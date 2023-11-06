import streamlit
import onprem
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from sentence_transformers import SentenceTransformer, util
import numpy as np


@streamlit.cache_resource
def load_llm(llm_config):
    return onprem.LLM(confirm=False, **llm_config)


@streamlit.cache_resource
def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def compute_similarity(sentence1, sentence2):
    model = get_embedding_model()
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.cpu().numpy()[0][0]


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + ""
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def setup_llm(llm_config):
    llm = load_llm(llm_config)
    chat_box = streamlit.empty()
    stream_handler = StreamHandler(chat_box, display_method="write")
    _ = llm.load_llm()
    llm.llm.callbacks = [stream_handler]
    return llm


def main():
    streamlit.set_page_config(
        page_title="Aarnos onpremllm demo", page_icon="🐍", layout="wide"
    )
    streamlit.title("Aarnos onpremllm demo")
    model = streamlit.sidebar.radio(
        "Model:",
        (
            "Wizard 7B",
            "Wizard 13B",
            "Falcon 7B",
            "Falcon 40B",
            "Mistral 7B",
            "OpenOrca 13B",
        ),
    )
    rag = streamlit.sidebar.checkbox("Use Retrieval Augmented Generation (RAG)")

    template = """
### Instruction:

{prompt}

### Response:"""

    if model == "Falcon 7B":
        model_config = {
            "model_url": "https://huggingface.co/hadongz/falcon-7b-instruct-gguf/resolve/main/falcon-7b-instruct-q4_0.gguf",
            "n_gpu_layers": 49,
        }

    elif model == "Falcon 40B":
        model_config = {
            "model_url": "https://huggingface.co/YokaiKoibito/falcon-40b-GGUF/resolve/main/falcon-40b-Q3_K_M.gguf",
            "n_gpu_layers": 49,
        }
    elif model == "OpenOrca 13B":
        model_config = {
            "model_url": "https://huggingface.co/kroonen/OpenOrca-Platypus2-13B-GGUF/resolve/main/OpenOrca-Platypus2-13B-Q4_K_M.gguf",
            "n_gpu_layers": 49,
        }
    elif model == "Mistral 7B":
        model_config = {
            "model_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
            "n_gpu_layers": 49,
        }
    elif model == "Wizard 7B":
        model_config = {
            "model_url": "https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF/resolve/main/Wizard-Vicuna-7B-Uncensored.Q4_K_M.gguf",
            "n_gpu_layers": 49,
        }
    elif model == "Wizard 13B":
        model_config = {
            "model_url": "https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q4_K_M.gguf",
            "n_gpu_layers": 49,
        }
    input = streamlit.text_input(
        "Submit a Prompt to the LLM:",
        placeholder="As a software developer, what is VSHN AppOps?",
    )

    if not rag:  # run model without RAG
        # submit_button = streamlit.button("Prompt")
        streamlit.markdown("---")
        llm = setup_llm(model_config)
        if input:
            print(input)
            saved_output = llm.prompt(input, prompt_template=template)
    else:  # run model with RAG
        # ask_button = streamlit.button("Ask")
        streamlit.markdown("---")
        llm = setup_llm(model_config)
        llm.ingest("./sample_data")
        if input:
            print(input)
            result = llm.ask(input)
            answer = result["answer"]
            docs = result["source_documents"]
            unique_sources = set()
            for doc in docs:
                answer_score = compute_similarity(answer, doc.page_content)
                question_score = compute_similarity(input, doc.page_content)
                if answer_score < 0.5 or question_score < 0.3:
                    continue
                unique_sources.add(
                    (
                        doc.metadata["source"],
                        doc.metadata.get("page", None),
                        doc.page_content,
                        question_score,
                        answer_score,
                    )
                )
            unique_sources = list(unique_sources)
            unique_sources.sort(key=lambda tup: tup[-1], reverse=True)
            if unique_sources:
                streamlit.markdown(
                    "**One or More of These Sources Were Used to Generate the Answer:**"
                )
                # streamlit.markdown(
                #     "*You can inspect these sources for more information and to also guard against hallucinations in the answer.*"
                # )
                for source in unique_sources:
                    fname = source[0]
                    # fname = construct_link(
                    #     fname, source_path=RAG_SOURCE_PATH, base_url=RAG_BASE_URL
                    # )
                    page = source[1] + 1 if isinstance(source[1], int) else source[1]
                    content = source[2]
                    question_score = source[3]
                    answer_score = source[4]
                    streamlit.markdown(
                        f"- {fname} {', page '+str(page) if page else ''} : score: {answer_score:.3f}",
                        help=f"{content}... (QUESTION_TO_SOURCE_SIMILARITY: {question_score:.3f})",
                        unsafe_allow_html=True,
                    )
            elif "I don't know" not in answer:
                streamlit.warning(
                    "No sources met the criteria to be displayed. This suggests the model may not be generating answers directly from your documents "
                    + "and increases the likelihood of false information in the answer. "
                    + "You should be more cautious when using this answer."
                )


if __name__ == "__main__":
    main()
