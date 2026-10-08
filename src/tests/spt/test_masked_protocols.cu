// Numerical diagnostics for SGFPT, not a deployment/dealer implementation.
// Each process emulates its dealer key generation using the same TEST seed.
// Online values are masked openings x+r; decoding subtracts the output mask.
#include "Server-SGFPT/sample.h"
#include "Server-SGFPT/select_top.h"
#include "Server-SGFPT/update.h"
#include "utils/gpu_file_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using Vec = std::vector<double>;
using Words = std::vector<u64>;
constexpr double FP = 16777216.0;
static int failures = 0, party = 0, test_round = 0;

Words copy_words(const u64* ptr, size_t n) {
    Words out(n);
    checkCudaErrors(cudaMemcpy(out.data(), ptr, n*sizeof(u64), cudaMemcpyDeviceToHost));
    return out;
}
void put_words(u64* ptr, const Words& x) {
    checkCudaErrors(cudaMemcpy(ptr, x.data(), x.size()*sizeof(u64), cudaMemcpyHostToDevice));
}
u64 encode(double x) { return static_cast<u64>(static_cast<int64_t>(std::llround(x*FP))); }
double decode(u64 x) { return static_cast<double>(static_cast<int64_t>(x))/FP; }
Words masked(const Vec& x, const Words& r) {
    Words out(x.size());
    for(size_t i=0;i<x.size();++i) out[i]=encode(x[i])+r[i];
    return out;
}
Words new_mask(u64* ptr, size_t n, bool zero) {
    auto tmp=randomGEOnGpu<u64>(n,64);
    auto out=copy_words(tmp,n);gpuFree(tmp);put_words(ptr,out);
    size_t nonzero=std::count_if(out.begin(),out.end(),[](u64 x){return x!=0;});
    if((zero && nonzero) || (!zero && nonzero!=n)) {
        fprintf(stderr,"Invalid test mask setup\n");std::exit(3);
    }
    return out;
}
void check(const char* label,const u64* actual,const Words& mask,const Vec& expected,double tolerance) {
    auto x=copy_words(actual,expected.size());
    double maxerr=0;size_t bad=0;double first=0,firstref=0;
    for(size_t i=0;i<x.size();++i){
        double value=decode(x[i]-mask[i]);double error=std::abs(value-expected[i]);
        if(!std::isfinite(value)||error>tolerance){if(!bad){first=value;firstref=expected[i];}++bad;}
        maxerr=std::max(maxerr,error);
    }
    printf("RESULT {\"party\":%d,\"round\":%d,\"field\":\"%s\",\"n\":%zu,\"bad\":%zu,\"max_abs_error\":%.12g,\"tolerance\":%.12g,\"first_bad_value\":%.12g,\"first_bad_expected\":%.12g}\n",party,test_round,label,x.size(),bad,maxerr,tolerance,first,firstref);
    fflush(stdout);failures+=bad!=0;
}

struct Reference { Vec m,sigma,C,pc,ps; };
// Independent double-precision diagonal CMA-ES equations (no GPU coefficients).
Reference update_reference(const Reference& old,const Vec& Y,const Vec& Z,int mu,int d) {
    Vec w(mu);double sw=0,ss=0;
    for(int i=0;i<mu;++i){w[i]=std::log(mu+0.5)-std::log(i+1.0);sw+=w[i];}
    for(double& v:w){v/=sw;ss+=v*v;}
    double eff=1/ss, cs=(eff+2)/(d+eff+5), cc=(4+eff/d)/(d+4+2*eff/d);
    double c1=2/(std::pow(d+1.3,2)+eff);
    double cmu=std::min(1-c1,2*(eff-2+1/eff)/(std::pow(d+2,2)+eff));
    Reference out=old;double normsq=0;
    for(int k=0;k<d;++k){
        double yw=0,zw=0,ysq=0;
        for(int i=0;i<mu;++i){yw+=w[i]*Y[i*d+k];zw+=w[i]*Z[i*d+k];ysq+=w[i]*Y[i*d+k]*Y[i*d+k];}
        out.m[k]=old.m[k]+old.sigma[k]*yw;
        out.ps[k]=(1-cs)*old.ps[k]+std::sqrt(cs*(2-cs)*eff)*zw;
        normsq+=out.ps[k]*out.ps[k];
        out.pc[k]=(1-cc)*old.pc[k]+std::sqrt(cc*(2-cc)*eff)*yw;
        out.C[k]=(1-c1-cmu)*old.C[k]+c1*out.pc[k]*out.pc[k]+cmu*ysq;
    }
    double en=std::sqrt(double(d))*(1-1.0/(4*d)+1.0/(21*d*d));
    double sigma=old.sigma[0]*std::exp(cs/(1+cs)*(std::sqrt(normsq)/en-1));
    std::fill(out.sigma.begin(),out.sigma.end(),sigma);return out;
}

int main(int argc,char** argv) {
    // party peer mode zero|nonzero seed generation holders
    if(argc!=8){fprintf(stderr,"Usage: %s party peer sample|select|update|chain|chain2 zero|nonzero seed generation holders\n",argv[0]);return 3;}
    setvbuf(stdout,nullptr,_IOLBF,0);
    party=std::stoi(argv[1]);std::string mode=argv[3],mask_mode=argv[4];
    bool zero=mask_mode=="zero";unsigned long long seed=std::stoull(argv[5]);
    int generation=std::stoi(argv[6]),holders=std::stoi(argv[7]);
    if((party!=0&&party!=1)||(mask_mode!="zero"&&mask_mode!="nonzero")||holders<1||holders>3||generation<0 ||
       (mode!="sample"&&mode!="select"&&mode!="update"&&mode!="chain"&&mode!="chain2")) return 3;
    const bool chain = mode=="chain" || mode=="chain2";
    const bool do_sample = mode=="sample" || chain;
    const bool do_select = mode=="select" || chain;
    const bool do_update = mode=="update" || chain;
    checkCudaErrors(cudaSetDevice(0));initGPUMemPool();AESGlobalContext aes;initAESContext(&aes);
    initGPURandomness();initCPURandomness();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gpuGen[0],seed));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(cpuGen[0],seed+1234567));
    setZeroRandomness(zero);
    constexpr int d=8,lambda=8,mu=4;
    // Match the service driver's precisions and LUT widths.
    CMAConfig cfg{d,lambda,mu,24,64,18,24,14,24,22,25};
    CMAState<u64> state(cfg);CMAMask<u64> masks(cfg);state.init();masks.init();
    for (auto pair : {std::make_pair(state.d_w,masks.d_w),
                      std::make_pair(state.d_csigma2_w,masks.d_csigma2_w),
                      std::make_pair(state.d_cmu_w,masks.d_cmu_w)}) {
        auto online=copy_words(pair.first,mu),offline=copy_words(pair.second,mu);
        if(online!=offline){fprintf(stderr,"Online/offline coefficient mismatch\n");return 3;}
        printf("COEFFICIENT online=[%llu,%llu,%llu,%llu] offline=[%llu,%llu,%llu,%llu]\n",
            (unsigned long long)online[0],(unsigned long long)online[1],(unsigned long long)online[2],(unsigned long long)online[3],
            (unsigned long long)offline[0],(unsigned long long)offline[1],(unsigned long long)offline[2],(unsigned long long)offline[3]);
    }

    Reference ref{Vec(d),Vec(d,0.75),Vec(d),Vec(d),Vec(d)};
    for(int k=0;k<d;++k){ref.m[k]=(k-3)*0.03125;ref.C[k]=1+(k%3)*0.125;ref.pc[k]=(k-4)*0.015625;ref.ps[k]=(k-3)*0.0078125;}
    // Snapshot and apply INPUT masks before keygen mutates CMAMask in place.
    put_words(state.d_masked_m,masked(ref.m,new_mask(masks.d_mask_m,d,zero)));
    put_words(state.d_masked_sigma,masked(ref.sigma,new_mask(masks.d_mask_sigma,d,zero)));
    put_words(state.d_masked_C_diag,masked(ref.C,new_mask(masks.d_mask_C_diag,d,zero)));
    put_words(state.d_masked_p_c,masked(ref.pc,new_mask(masks.d_mask_p_c,d,zero)));
    put_words(state.d_masked_p_sigma,masked(ref.ps,new_mask(masks.d_mask_p_sigma,d,zero)));
    Vec Y(lambda*d),Z(lambda*d),selectedY(mu*d),selectedZ(mu*d);
    for(int i=0;i<lambda*d;++i){Y[i]=(i-29)*0.015625;Z[i]=(i%13-6)*0.03125;}
    put_words(state.d_masked_Y,masked(Y,new_mask(masks.d_mask_Y,lambda*d,zero)));
    // Sample generates public Z; its input mask is therefore zero in SelectTop.
    put_words(masks.d_mask_Z,Words(lambda*d));put_words(state.d_Z,masked(Z,Words(lambda*d)));
    for(int i=0;i<mu*d;++i){selectedY[i]=(i-13)*0.015625;selectedZ[i]=(i%9-4)*0.03125;}
    put_words(state.d_masked_mu_Y,masked(selectedY,new_mask(masks.d_mask_mu_Y,mu*d,zero)));
    put_words(state.d_masked_mu_Z,masked(selectedZ,new_mask(masks.d_mask_mu_Z,mu*d,zero)));
    // Unique candidate losses and nonidentical per-holder contributions.
    const int order_values[lambda]={4,1,6,3,0,7,2,5};Vec A(lambda*holders);Vec losses(lambda,0);
    for(int j=0;j<lambda;++j) for(int h=0;h<holders;++h){A[j*holders+h]=(order_values[j]+1)*(h+1)*0.03125;losses[j]+=A[j*holders+h];}
    std::vector<int> order(lambda);std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){return losses[a]<losses[b];});
    spt::Sample<u64> sample;spt::Update<u64> update;
    spt::SelectTopParams params{lambda,mu,d,holders,24,28,64,8};spt::SelectTop<u64> select(params);
    u8* base=nullptr;u8* cursor=nullptr;getKeyBuf(&base,&cursor,OneGB);
    GpuPeer peer(true);peer.connect(party,argv[2]);
    if(do_select)select.init();
    if(do_update)update.init(state);
    Words previous_z;
    for(test_round=0;test_round<(mode=="chain2"?2:1);++test_round){
    const int current_generation=generation+test_round;
    setZeroRandomness(zero);
    cursor=base;
    auto maskA=randomGEOnGpu<u64>(lambda*holders,64);
    auto holder_masks=copy_words(maskA,lambda*holders);
    auto inputA=masked(A,holder_masks);
    auto deviceA=(u64*)gpuMalloc(inputA.size()*sizeof(u64));put_words(deviceA,inputA);
    gpuFree(maskA);
    // Dealer supplies one mask per aggregated candidate, as in test_select_top.
    Words aggregate_masks(lambda,0);
    for(int j=0;j<lambda;++j)for(int h=0;h<holders;++h)
        aggregate_masks[j]+=holder_masks[j*holders+h];
    Words ry,rx,rsy,rsz,rm,rs,rc,rpc,rps;
    printf("CASE mode=%s mask=%s seed=%llu generation=%d holders=%d party=%d\n",mode.c_str(),mask_mode.c_str(),seed,current_generation,holders,party);
    if(do_sample){
        sample.keygen(&cursor,party,masks,&aes,current_generation);ry=copy_words(masks.d_mask_Y,lambda*d);rx=copy_words(masks.d_mask_X,lambda*d);
    }
    if(do_select){
        auto aggregateMaskA=(u64*)gpuMalloc(lambda*sizeof(u64));put_words(aggregateMaskA,aggregate_masks);
        select.keygen(&cursor,party,64,64,aggregateMaskA,masks,&aes);
        rsy=copy_words(masks.d_mask_mu_Y,mu*d);rsz=copy_words(masks.d_mask_mu_Z,mu*d);
    }
    if(do_update){
        update.keygen(&cursor,party,masks,&aes);
        rm=copy_words(masks.d_mask_m,d);rs=copy_words(masks.d_mask_sigma,d);rc=copy_words(masks.d_mask_C_diag,d);
        rpc=copy_words(masks.d_mask_p_c,d);rps=copy_words(masks.d_mask_p_sigma,d);
    }
    printf("KEYGEN bytes=%zu\n",size_t(cursor-base));
    u8* read=base;
    if(do_sample)sample.readkey(&read,masks);
    if(do_select)select.readkey(&read);
    if(do_update)update.readkey(&read,state);
    if(read!=cursor){fprintf(stderr,"Key buffer parsing mismatch\n");return 3;}
    setZeroRandomness(false);
    if(do_sample){
        // Reusing a different generation's key must fail before online work.
        bool rejected=false;
        try{sample.run(&peer,party,state,&aes,nullptr,current_generation+1);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected){fprintf(stderr,"Sample accepted a mismatched generation key\n");return 3;}
        sample.run(&peer,party,state,&aes,nullptr,current_generation);
        auto z=copy_words(state.d_Z,lambda*d);Vec X(lambda*d);
        if(!previous_z.empty()&&z==previous_z){fprintf(stderr,"Sample reused Z across generations\n");return 3;}
        previous_z=z;
        for(int i=0;i<lambda*d;++i){Z[i]=decode(z[i]);Y[i]=std::sqrt(ref.C[i%d])*Z[i];X[i]=ref.m[i%d]+ref.sigma[i%d]*Y[i];}
        check("sample.Y",state.d_masked_Y,ry,Y,1e-4);check("sample.X",state.d_masked_X,rx,X,1e-4);
    }
    if(do_select){
        select.run(&peer,party,&read,64,64,deviceA,state,&aes,nullptr);
        for(int i=0;i<mu;++i)for(int k=0;k<d;++k){selectedY[i*d+k]=Y[order[i]*d+k];selectedZ[i*d+k]=Z[order[i]*d+k];}
        double tol=chain?1e-4:0;
        check("select.Y",state.d_masked_mu_Y,rsy,selectedY,tol);check("select.Z",state.d_masked_mu_Z,rsz,selectedZ,tol);
    }
    if(do_update){
        update.run(&peer,party,state,&aes,nullptr);
        auto expected=update_reference(ref,selectedY,selectedZ,mu,d);
        check("update.m",state.d_masked_m,rm,expected.m,5e-4);
        check("update.sigma",state.d_masked_sigma,rs,expected.sigma,5e-4);
        check("update.C",state.d_masked_C_diag,rc,expected.C,5e-4);
        check("update.pc",state.d_masked_p_c,rpc,expected.pc,5e-4);
        check("update.ps",state.d_masked_p_sigma,rps,expected.ps,5e-4);
        // Keep actual masked state and dealer masks; only advance the CPU reference.
        ref=expected;
    }
    gpuFree(deviceA);
    peer.sync();
    }
    peer.close();
    printf("SUMMARY failed_fields=%d\n",failures);return failures?1:0;
}
