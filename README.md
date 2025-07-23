# Friday AI Assistant - Your Personal JARVIS

## Project Overview
Friday is a personalized AI assistant inspired by Iron Man's JARVIS, fine-tuned for Telugu film production and cooking domains. This project aims to create a custom AI that understands your specific needs and responds to voice commands like "Daddy's home!"

## Project Architecture

### Phase 1: Data Collection & Preparation
- **Production Domain Data**: Telugu film industry knowledge, production workflows, casting, scheduling
- **Cooking Domain Data**: Recipes, cooking techniques, ingredient knowledge
- **Personal Assistant Data**: Daily tasks, preferences, voice interaction patterns
- **Voice Command Data**: Custom wake words and responses

### Phase 2: Model Fine-tuning
- Base model: Llama 2 (7B or 13B parameter model)
- Fine-tuning approach: LoRA (Low-Rank Adaptation) for efficiency
- Training data: Custom datasets for each domain
- Evaluation: Domain-specific metrics and personal assistant capabilities

### Phase 3: Deployment & Infrastructure
- Local server setup with GPU support
- Web interface for employee access
- Voice interface with wake word detection
- REST API for integration

## Directory Structure
```
friday/
├── data/                    # Training and evaluation data
│   ├── production/         # Telugu film industry data
│   ├── cooking/           # Cooking domain data
│   ├── assistant/         # Personal assistant data
│   └── voice/             # Voice command data
├── models/                 # Model files and checkpoints
├── training/              # Training scripts and configurations
├── inference/             # Inference and deployment code
├── web/                   # Web interface
├── voice/                 # Voice processing components
└── docs/                  # Documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: RTX 4090 or better)
- 32GB+ RAM
- 1TB+ SSD storage

### Installation
```bash
# Activate virtual environment
source friyay/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Next Steps
1. Set up data collection pipeline
2. Download and prepare base Llama model
3. Create custom datasets
4. Implement fine-tuning pipeline
5. Build voice interface
6. Deploy to production server

## Voice Commands
- "Daddy's home!" - Wake command
- "Friday, help me with production schedule"
- "Friday, what's for dinner tonight?"
- "Friday, remind me about the casting call"

## License
This project is for personal use and Telugu film industry applications. 

# Download gguf model:
brew install aria2c

mkdir -p models/gguf
cd models/gguf
# quickest, resumable download
aria2c -x 8 \
  https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf
cd ../../         # back to repo root


# installed sentencepiece , sentence transformers through conda and it worked fine, changed the gguf filename manually to Meta-Llama-3-8B-Instruct.Q4_0.gguf  , 

# This syntax of running server worked. 

# wrong method:   
--n-gpu-layers -1 \
  --chat-format chatml \

# Right method:

python -m llama_cpp.server \
  --model models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --n_gpu_layers -1 \
  --chat_format chatml \
  --port 8000