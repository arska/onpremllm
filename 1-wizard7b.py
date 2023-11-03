from onprem import LLM
import sys

if __name__ == '__main__':
    llm = LLM(use_larger=True, n_gpu_layers=49)

    if (len(sys.argv) == 1):
        prompt = """What is VSHN?"""
        saved_output = llm.prompt(prompt)
        print(saved_output)
