import streamlit as st
import openai
import pinecone

st.header("Mr. Edge")
input_txt = st.text_input("Ask a question")

if st.button("Send"):
    query = input_txt

    openai.api_key = st.secrets['openai_api_key']  #platform.openai.com

    embed_model = "text-embedding-ada-002"

    index_name = 'edge-user'

    # initialize connection to pinecone
    pinecone.init(
        api_key = st.secrets['pinecone_api_key'],  # app.pinecone.io (console)
        environment="eu-west1-gcp"  # next to API key in console
    )
    # connect to index
    index = pinecone.GRPCIndex(index_name)


    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True, namespace='langchain-chunking')


    contexts = [item['metadata']['text'] for item in res['matches']]
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query


    primer = f"""You are a customer support agent. A highly intelligent agent that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know". Respond using markdown.
    """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.2,
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ],
        stream = True
    )

    container = st.empty()
    messages: list[str] = []
    for chunk in res:
        chunk_message = chunk['choices'][0]['delta'].get('content', '')
        messages.append(chunk_message)
        container.markdown("".join(messages))

    # st.markdown(res['choices'][0]['message']['content'])
