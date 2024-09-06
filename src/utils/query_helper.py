# src/utils/query_utils.py
from llama_index.core import StorageContext, load_index_from_storage, ChatPromptTemplate

def handle_query(query, persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
        ("user", """You are a Q&A assistant named CHATTO, created by Rishi. You have a specific response programmed for when users specifically ask about your creator, Rishi. The response is: "I was created by Rishi, an enthusiast in Artificial Intelligence. He is dedicated to solving complex problems and delivering innovative solutions. With a strong focus on machine learning, deep learning, Python, generative AI, NLP, and computer vision, Rishi is passionate about pushing the boundaries of AI to explore new possibilities." For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
        Context:
        {context_str}
        Question:
        {query_str}
        """)
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."