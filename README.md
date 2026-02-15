# what-is-missing
The code for the paper "What Is Missing: Interpretable Ratings For Large Language Model Outputs. The paper can be found here.

## Models

To work with the models in this project, ensure you're using `git lfs` (Git Large File Storage) for efficient model downloading. Here’s how to set up and retrieve the models:

1. **Install Git LFS**  
   ```bash
   module load git-lfs/3.4.0
   git lfs install
   ```

2. **Pull Models**  
   Navigate into the desired directory and use the following commands to pull the models:

   - **Mixtral 8x7B Instruct (with Flash Attention)**  
     - URL: [Mixtral 8x7B Instruct v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main)  
     - Clone:  
       ```bash
       git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
       ```

   - **Llama3 8B Instruct**  
     - URL: [Llama3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main)  
     - Clone:  
       ```bash
       git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
       ```

    - **all-mpnet-base-v2 (Sentence Transformer)**  
      - URL: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main)  
      - Clone:  
       ```bash
       git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
       ```

## Required Libraries

All dependencies are listed in the `requirements.txt`. The key libraries include:

- Huggingface `transformers`
- `Pytorch`
- Flash Attention (Only on A100 GPUs, not included in current implementation)
- Huggingface `trl`
- Sentence Transformers
- Comet.ml

To install the required packages:
```bash
pip install --no-index transformers
pip install --no-index torch
pip install --no-index bitsandbytes
pip install --no-index -U flash-attn --no-build-isolation (Only on A100 GPUs)
pip install --no-index -U sentence-transformers
pip install --no-index trl
```

## GPU RAM Calculation

On the Beluga cluster, each node provides 64GB of GPU RAM, 16GB in each GPU. By using quantization with 4-bit parameters, the Mixtral model will require approximately 27GB of GPU memory. Llama 3-8B is listed to require 16Gb of GPU memory. Running both models requires 43GB. I found that there was just under the required amount of memory with 3 GPUs. I requested 4 GPUs for a total of 64GB GPU RAM. RAM is also required for the sentence transformer model but 64GB should still be sufficient.

NOTE: This does not include any extra overhead for the training models. The numbers listed above are strictly for inference.


## Running on a Cluster

To run the models on a cluster, use the following command:
```bash
sbatch llm-interaction-job.sh
```

To run the Online DPO trainer on a cluster, use the following command:
```bash
sbatch odpo-trainer-job.sh
```

Make sure all file paths in the script are correctly set for the cluster's file system.

## Note on Generation

I am using contrastive search as seen in this [blog post](https://huggingface.co/docs/transformers/en/generation_strategies). The parameters are automatically set to defaults but can be changed for tasks like topic generation. Alternatively, high temperature sampling can produce better results in terms of training.

## Online DPO

To stay within the Hugging Face toolset, I will be using the TRL library found [here](https://huggingface.co/docs/trl/index).

## Datasets

For initial testing I am using the ultrafeedback-prompt dataset. It is a standard conversational dataset that is used in the TRL examples. Remember to use git lfs.

- URL: [ultrafeedback-prompt](https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt)
- Clone:  
 ```bash
 git clone https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt
 ```

## Sentence Transformers

Producing useful embeddings is important to actually tell the model what knowledge is missing. I am using [Sentence Transformers](https://huggingface.co/sentence-transformers) to extract the useful embeddings. These are usually used for semantic search and is more useful than the standard LLM tokenizers.

## Helpful and Similar Papers

- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267)
- [ETA-REWARDING LANGUAGE MODELS: Self-Improving Alignment with LLM-as-a-Meta-Judge](https://arxiv.org/pdf/2407.19594)
- [Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://arxiv.org/pdf/2312.01823)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685)
- [Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020)
- [WEAK-TO-STRONG GENERALIZATION: ELICITING STRONG CAPABILITIES WITH WEAK SUPERVISION](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)
