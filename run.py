from onprem import LLM
import sys

if __name__ == "__main__":
    url = sys.argv[1]
    llm = LLM(url, n_gpu_layers=999)
    answer = llm.prompt("As a software developer, what is VSHN AppOps?")
