# Quality Assessor (based on Alpaca_EVAL)

**How to run the assessor:** 
1. install the alpaca-eval:
   
```bash
pip install -e .
```
if you install alpaca-eval before, you need first uninstall the alpaca-eval.

2. Setup Open-AI API key.
```bash
export OPENAI_API_KEY=<your_api_key>
```
3. Begin your testing.
```bash
alpaca_eval evaluate --input_path=<INPUT_PATH> --output_path=<OUTPUT_PATH> --annotators_config=YOUR_ANNOTATOR
```
