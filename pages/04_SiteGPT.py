
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from urllib.parse import urlparse
import streamlit as st
import os


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

if 'api_key' not in st.session_state:
    st.markdown(
        """
        # SiteGPT
                
        Ask questions about cloudflare.
                
        Start by putting the API KEY on the sidebar.
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
    st.markdown(
        """
        # SiteGPT
                
        """
    )
        
    openai_api_key = st.session_state['api_key']

    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=openai_api_key,
    )

    answers_prompt = ChatPromptTemplate.from_template(
        """
        Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
        Then, give a score to the answer between 0 and 5.

        If the answer answers the user question the score should be high, else it should be low.

        Make sure to always include the answer's score even if it's 0.

        Context: {context}
                                                    
        Examples:
                                                    
        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
                                                    
        Question: How far away is the sun?
        Answer: I don't know
        Score: 0
                                                    
        Your turn!

        Question: {question}
    """
    )


    def get_answers(inputs):
        docs = inputs["docs"]
        question = inputs["question"]
        answers_chain = answers_prompt | llm

        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, "context": doc.page_content}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
        }


    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use ONLY the following pre-existing answers to answer the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
                """,
            ),
            ("human", "{question}"),
        ]
    )


    def choose_answer(inputs):
        answers = inputs["answers"]
        question = inputs["question"]
        choose_chain = choose_prompt | llm
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke(
            {
                "question": question,
                "answers": condensed,
            }
        )


    def parse_page(soup):
        header = soup.find("header")
        footer = soup.find("footer")
        if header:
            header.decompose()
        if footer:
            footer.decompose()
        return (
            str(soup.get_text())
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("CloseSearch Submit Blog", "")
        )

    def extract_site_name(url):
        parsed_url = urlparse(url)
        return parsed_url.netloc


    @st.cache_data(show_spinner="Loading website...")
    def load_website(url):
        dir_path = f"./.cache/embeddings_sites/{extract_site_name(url)}"
        
        # 폴더가 존재하는지 확인하고, 없으면 생성
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        cache_dir = LocalFileStore(dir_path)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )

        filters = [
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ]

        loader = SitemapLoader(
            url,
            parsing_function=parse_page,
            filter_urls=filters,
        )

        loader.requests_per_second = 2
        docs = loader.load_and_split(text_splitter=splitter)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()


    url="https://developers.cloudflare.com/sitemap-0.xml"
    retriever = load_website(url)
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
