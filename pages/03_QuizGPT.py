import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
import os




st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

if 'api_key' not in st.session_state:
    st.markdown(
    """
    Welcome!
                
    This GPT generate quiz base on your keyword or file file!

    ##### On the sidebar, Type You 'OPEN API KEY'
    """
    )
    with st.sidebar:
        api_key = st.text_input("Enter your key:")

        if st.button("Submit API Key"):
            if api_key:
                st.session_state['api_key'] = api_key
                st.success("API key saved!")
                st.experimental_rerun()
            else:
                st.error("Please enter a valid API key.")

else:
    openai_api_key = st.session_state['api_key']

    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )


    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    def get_difficulty(difficulty):
        return difficulty

    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.

        and Adjust the difficulty of the quiz according to the difficulty level given below.

        Context: {context},
        Difficulty: {difficulty}
    """,
            )
        ]
    )

    questions_chain = {"context": format_docs, "difficulty": get_difficulty} | questions_prompt | llm


    @st.cache_data(show_spinner="Loading file...")
    def split_file(file):
        # file_content = file.read()
        # file_path = f"./.cache/quiz_files/{file.name}"

        save_dir = "./.cache/files/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        return docs


    @st.cache_data(show_spinner="Making quiz...")
    def run_quiz_chain(_docs, topic):
        chain = questions_chain
        return chain.invoke(_docs)


    @st.cache_data(show_spinner="Searching Wikipedia...")
    def wiki_search(term):
        retriever = WikipediaRetriever(top_k_results=5)
        docs = retriever.get_relevant_documents(term)
        return docs


    with st.sidebar:
        docs = None
        topic = None
        difficulty = st.selectbox(
            "Choose difficulty of test",
            (
                "Hard",
                "Easy",
            )
        )
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)


    if not docs:
        st.markdown(
            """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
        )
    else:
        message_chunk = run_quiz_chain(docs, topic if topic else file.name)
        response = json.loads(message_chunk.additional_kwargs['function_call']['arguments'])
        with st.form("questions_form"):
            correct_answers_count = 0
            for i, question in enumerate(response["questions"]):
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"quesion_{i}"
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    correct_answers_count += 1
                elif value is not None:
                    st.error("Wrong!")
            if correct_answers_count >= 10 :
                st.balloons()
                st.success("Congratulations! 🎉 You got all the questions!")
            
            button = st.form_submit_button(disabled=(correct_answers_count==10))