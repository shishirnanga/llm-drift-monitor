# LLM Performance Comparison Study

**Comprehensive analysis of 6 major AI models across 12,870 test results over 8 weeks**

## Key Findings

- GPT-4o: 95.4% accuracy (best overall)
- GPT-3.5 Turbo: Perfect at math (100%) at 1/10th the cost
- Gemini Pro: Blocked 100% of coding prompts
- Claude: Best for creative writing (80%)
- No significant drift detected across any model

## Models Tested

- GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo (OpenAI)
- Claude Sonnet 4 (Anthropic)
- Gemini 1.5 Pro, Gemini 1.5 Flash (Google)

## Test Categories (65 tests)

- Mathematical reasoning
- Logical reasoning  
- Creative writing
- Code generation
- Factual knowledge
- Instruction following
- Consistency

## Results

See [PUBLICATION.md](PUBLICATION.md) for full analysis.

## Data

- 33 test runs over 8 weeks
- 12,870 individual test results
- All raw data in `data/raw/`
- Statistical analysis with t-tests and effect sizes

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key

# Run tests
python main.py

# Analyze drift
python analyze.py drift

# View dashboard
streamlit run dashboard/app.py
```

## Citation

If you use this work, please cite:
```
Nanga, S. (2026). Comparative Analysis of Six Large Language Models.
Independent Research. https://github.com/shishirnanga/llm-drift-monitor
```

## Contact

www.linkedin.com/in/shishir-nanga