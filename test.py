from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(temperature=0.1)

template = PromptTemplate.from_template(
    "What is the distance between {country_a} and {country_b}"
)
prompt = template.format(country_a="Mexico", country_b="Thailand")

chat.predict(prompt)


template = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert. And you only reply in {language}."),
    ("ai", "Ciao, mi chiamo {name}!"),
    ("human", "What is the distance between {country_a} and {country_b}. Also, what is your name?")
])

prompt = template.format_messages(
    language="Greek",
    name="Socrates",
    country_a="Mexico",
    country_b="Thailand"
)

chat.predict_messages(prompt)

from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):

    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))

p = CommaOutputParser()
p.parse("Hello,how,are, you?")

template = ChatPromptTemplate.from_messages([
    ("system", "You are a list generating machine. Everything you are asked will be answered with a comma separated list of max {max_items} in lowercase. Do NOT reply with anything else."),
    ("human", "{question}"),
])

# prompt = template.format_messages(
#     max_items=10,
#     question="What are the colors?"
# )

# result = chat.predict_messages(prompt)

# p = CommaOutputParser()

# p.parse(result.content)

chain = template | chat | CommaOutputParser()

chain.invoke({"max_items": 5, "question": "What are the pokemons?"})