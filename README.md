# Street Fighter III AI Agent - DIAMBRA Competition

A reinforcement learning AI agent trained to play Street Fighter III: 3rd Strike using PPO (Proximal Policy Optimization) and curriculum learning. This project achieves competitive performance through a 3-phase training pipeline that progressively increases difficulty.

Came top 10 in a global competition for training street fighter agents

## Project Overview

This AI agent uses **curriculum learning** to master Street Fighter III by training across three difficulty phases:
- **Phase 1**: Easy difficulty (Difficulty 1) - Foundation learning
- **Phase 2**: Normal difficulty (Difficulty 2) - Skill development
- **Phase 3**: Hard difficulty (Difficulty 3) - Expert performance

**Final Performance**: 60%+ win rate against difficulty 3 opponents

## Key Features

- **PPO Algorithm** from Stable Baselines3
- **Curriculum Learning** across 3 difficulty phases
- **Frame Stacking** (4 frames) for temporal awareness
- **Custom Reward Shaping** for combat effectiveness
- **Grayscale 84x84** observations for efficiency
- **Multi-character Support**: Ken, Ryu, Chun-Li
- **~10 million** total training steps

## Quick Start

### Prerequisites

- Python 3.12+
- DIAMBRA Arena (requires Docker)
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sfiii-ai-agent.git
cd sfiii-ai-agent
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install diambra-arena[stable-baselines3]
pip install stable-baselines3 torch numpy
```

4. Download the trained model from Hugging Face:
```bash
# Install huggingface_hub
pip install huggingface_hub

# Download the model
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='YOUR_USERNAME/sfiii-agent', filename='model.zip', local_dir='./models')"
```

### Running the Agent

```bash
# Test the trained agent
diambra run python test_agent.py

# Or run the competition submission agent
diambra run python models_phase3/agent.py
```

## Training from Scratch

### Phase 1: Easy Difficulty
```bash
caffeinate -d diambra run -s=4 python train_phase1.py
```
- **Duration**: ~2M steps
- **Time**: ~6-8 hours on RTX 3080
- **Saves to**: `./models_phase1/`

### Phase 2: Normal Difficulty
```bash
caffeinate -d diambra run -s=4 python train_phase2_correct.py
```
- **Duration**: ~2.88M steps
- **Time**: ~8-12 hours
- **Loads**: Phase 1 checkpoint
- **Saves to**: `./models_phase2/`

### Phase 3: Hard Difficulty
```bash
caffeinate -d diambra run -s=4 python train_phase3.py
```
- **Duration**: ~5M steps
- **Time**: ~18-24 hours
- **Loads**: Phase 2 checkpoint
- **Saves to**: `./models_phase3/`

### Resume Training from Checkpoint
```bash
# Resume Phase 3 training
python resume_phase3.py

# Or resume Phase 2
python resume_training.py
```

## Project Structure

```
├── train_phase1.py           # Phase 1 training script
├── train_phase2_correct.py   # Phase 2 training script
├── train_phase3.py           # Phase 3 training script (final)
├── train_elite.py            # Elite training configuration
├── train_ultimate.py         # Complete end-to-end pipeline
├── resume_phase3.py          # Resume Phase 3 from checkpoint
├── resume_training.py        # Resume Phase 2 from checkpoint
├── custom_rewards.py         # Custom reward shaping logic
├── test_agent.py             # Test trained agent
├── validate_submission.py    # Validate competition submission
├── diagnose.py               # Environment diagnostics
├── check_obs.py              # Observation space checker
├── models_phase3/
│   ├── agent.py              # Competition submission agent
│   ├── requirements.txt      # Submission dependencies
│   ├── Dockerfile            # Container configuration
│   └── README.md             # Submission guide
└── docs/
    ├── WINNING_STRATEGIES.md      # Advanced tactics
    ├── MODEL_ANALYSIS.md          # Performance analysis
    ├── QUICK_START_ELITE.md       # Fast training guide
    ├── SUBMISSION_GUIDE.md        # Competition submission
    ├── ADVANCED_TECHNIQUES.md     # Training techniques
    └── TRAINING_COMPARISON.md     # Training comparisons
```

## Model Architecture

### Network Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MultiInputPolicy
- **Network Architecture**: 512x512 hidden layers (2 layers)
- **Activation**: ReLU

### Hyperparameters
```python
learning_rate = 3e-4 (linear decay to 1e-5)
n_steps = 2048
batch_size = 256
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

### Observation Space
- **Frame size**: 84x84 grayscale
- **Frame stack**: 4 frames
- **Additional features**: Health, timer, stage info
- **Normalization**: Reward normalization enabled
- **Total dimensions**: Flattened observation vector

### Action Space
- **No attack button combinations** (simplified)
- **Step ratio**: 6 (action every 6 frames)

## Performance Metrics

| Phase | Difficulty | Training Steps | Win Rate | Avg Reward |
|-------|-----------|----------------|----------|------------|
| Phase 1 | 1 (Easy) | 2.0M | ~80% | ~3.5 |
| Phase 2 | 2 (Normal) | 2.88M | ~70% | ~2.8 |
| Phase 3 | 3 (Hard) | 5.67M | **60%+** | **2.14** |

## 🤗 Trained Model

The trained model is hosted on **Hugging Face** due to file size constraints:

### **[Download Model on Hugging Face →](https://huggingface.co/mmesomaa/streetfighter_agent)**

Replace `YOUR_USERNAME/sfiii-agent` with your actual Hugging Face repository URL.

### Quick Download

```python
from huggingface_hub import hf_hub_download

# Download the final model (20 MB)
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/sfiii-agent",
    filename="model.zip",
    local_dir="./models"
)

# Load and use
from stable_baselines3 import PPO
agent = PPO.load(model_path)
```

## Competition Submission

The agent is ready for DIAMBRA competition submission:

```bash
cd models_phase3
diambra run python agent.py
```

See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for detailed submission instructions.

## Training Tips

1. **Use GPU**: Training on CPU is ~10x slower
2. **Use `caffeinate -d`** on macOS to prevent sleep during long training
3. **Monitor with TensorBoard**: `tensorboard --logdir ./logs`
4. **Save checkpoints frequently**: Every 200k steps recommended
5. **Test checkpoints**: Use `test_agent.py` to evaluate performance
6. **Curriculum is key**: Don't skip phases - each builds on the previous

## Troubleshooting

### DIAMBRA Installation Issues
```bash
# Fix broken installation
source venv/bin/activate
pip uninstall -y diambra diambra-arena diambra-engine
pip install diambra-arena[stable-baselines3]

# Verify
python diagnose.py
```

### Common Errors

**Error**: `cannot import name 'SpaceTypes' from 'diambra.engine'`
**Solution**: Reinstall diambra-arena (see above)

**Error**: Model file not found
**Solution**: Download from Hugging Face or verify model path

**Error**: CUDA out of memory
**Solution**: Reduce `n_steps` or `batch_size` in training script

## Advanced Features

- **Custom Reward Shaping**: See [custom_rewards.py](custom_rewards.py)
- **Winning Strategies**: See [WINNING_STRATEGIES.md](docs/WINNING_STRATEGIES.md)
- **Advanced Techniques**: See [ADVANCED_TECHNIQUES.md](docs/ADVANCED_TECHNIQUES.md)
- **Model Analysis**: See [MODEL_ANALYSIS.md](docs/MODEL_ANALYSIS.md)

## Requirements

- Python 3.12+
- diambra-arena==2.2.7
- stable-baselines3
- torch (with CUDA support recommended)
- numpy
- Docker (for DIAMBRA)

## Citation

If you use this project in your research or competition, please cite:

```bibtex
@misc{sfiii-ai-agent,
  author = {Your Name},
  title = {Street Fighter III AI Agent using PPO and Curriculum Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/sfiii-ai-agent}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **DIAMBRA Arena**: Fighting game RL framework
- **Stable Baselines3**: PPO implementation
- **OpenAI Gym**: Environment interface
- Street Fighter III: 3rd Strike by Capcom

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Note**: This agent is for educational and competition purposes only. Street Fighter III is a trademark of Capcom.
