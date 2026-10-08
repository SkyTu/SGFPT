"""Frozen CLIP prompt evaluator, derived from the B2TPT integration."""
import torch
import numpy as np
import clip
from model.shallow_encoder import TextEncoder, VisionEncoder, VisionEncoder_Clip


class PromptCLIP_Shallow:
    def __init__(self, task_name, cfg, classes, n_cls):
        self.task_name = task_name
        self.opt_name = cfg["opt_name"]
        self.data_dir = cfg["data_dir"]
        self.output_dir = cfg["output_dir"]
        self.backbone = cfg["backbone"]
        self.popsize = cfg["popsize"]
        self.parallel = cfg["parallel"]
        self.batch_size = cfg["batch_size"]
        self.classes = classes
        self.n_cls = n_cls
        self.seed = cfg["seed"]
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(self.backbone, device=self.device)
        self.loss = []
        self.acc = []
        self.n_prompt_tokens_L = cfg["n_prompt_tokens_L"]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.ctx_dim_L = self.model.ln_final.weight.shape[0]
        self.text_encoder = TextEncoder(self.model)

        self.n_prompt_tokens_V = cfg["n_prompt_tokens_V"]
        self.ctx_dim_V = self.model.visual.width
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.image_encoder = VisionEncoder(self.model)
        self.image_encoder_clip = VisionEncoder_Clip(self.model)
        self.image_encoder.n_prompt_tokens_V = self.n_prompt_tokens_V

        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0
        self.min_loss = None
        self.pl_acc = None
        self.loss = []
        self.test_every = cfg["test_every"] if self.parallel else cfg["test_every"] * self.popsize
        self.sigma = cfg["sigma"]
        self.linear_L = torch.nn.Linear(self.intrinsic_dim_L, self.n_prompt_tokens_L * self.ctx_dim_L,
                                        bias=False, device=self.device, dtype=self.dtype)
        embedding = self.model.token_embedding.weight.cpu()
        mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        mu = 0.0
        projection_scale = cfg.get("projection_scale", 1.0)
        std = 1.0
        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {} | scale: {}x'.format(mu_hat, std_hat, mu, std, projection_scale))
        
        for p in self.linear_L.parameters():
            torch.nn.init.normal_(p, mu, std)
        self.linear_V = torch.nn.Linear(self.intrinsic_dim_V, self.n_prompt_tokens_V * self.ctx_dim_V,
                                        bias=False, device=self.device, dtype=self.dtype)
        conv = self.model.visual.conv1.weight.cpu()
        mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(conv.reshape(-1).detach().cpu().numpy())
        mu = mu_hat * 3072 / self.intrinsic_dim_V
        projection_scale = cfg.get("projection_scale", 1.0)
        std = 1.0
        print('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {} | scale: {}x'.format(mu_hat, std_hat, mu, std, projection_scale))
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)


    def get_text_information(self, caption=None):
        prompt_prefix = " ".join(["X"] * self.n_prompt_tokens_L)
        if caption is None:
            classnames = [name.replace("_", " ").replace("-", " ") for name in self.classes]
            pattern_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls": self.n_cls, "n_prompt_tokens_L": self.n_prompt_tokens_L,
                       "init_pattern_embedding": init_pattern_embedding,
                       "tokenized_pattern_prompts": tokenized_pattern_prompts,
                       "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        else:
            pattern_prompt = prompt_prefix + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls": 1, "n_prompt_tokens_L": self.n_prompt_tokens_L,
                       "init_pattern_embedding": init_pattern_embedding,
                       "tokenized_pattern_prompts": tokenized_pattern_prompts, "batch_size": self.batch_size,
                       "pop_size": self.popsize, "parallel": self.parallel}
        return context


    def get_image_information(self):
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context


    def generate_text_prompts(self, intrinsic_vectors):
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, -1)
            if self.init_prompt is not None:
                z = z + self.init_prompt  # Az + p_0

            prompt_list.append(z)
        return prompt_list


    def generate_visual_prompts(self, intrinsic_vectors):
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)  ## z 500
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V, -1)  ## z: 5,768
            visual_prompt_list.append(z)

        return visual_prompt_list


    @torch.no_grad()
    def eval(self, prompt_zip, batch, ii, r):
        prompt_text, prompt_image = prompt_zip[0], prompt_zip[1]
        self.num_call += 1
        loss = 0

        text_features = self.text_encoder(prompt_text)  # if parallel, text_features.shape = [n_cls * popsize, *, *]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image, label = self.parse_batch(batch)

        image_features = self.image_encoder(image, prompt_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = torch.mean(loss_fn(logits, label))



        correct = 0.
        prediction = logits.argmax(dim=-1)
        correct += (prediction == label).float().sum()
        acc = correct / int(label.shape[0])

        self.acc.append(acc)
        self.best_accuracy = max(acc, self.best_accuracy)
        return loss.item(), acc.item()


    def parse_batch(self, batch):
        image = batch["image"]
        label = batch["label"]
        image = image.to(device=self.device, dtype=self.dtype)
        label = label.to(device=self.device)

        return image, label


