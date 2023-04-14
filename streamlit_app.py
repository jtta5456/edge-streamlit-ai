import streamlit as st
import openai
import pinecone

st.header("Mr. Edge V2.2")
input_txt = st.text_input("Ask a question")

if st.button("Send"):
    query = input_txt

    openai.api_key = st.secrets['openai_api_key']  # platform.openai.com

    embed_model = "text-embedding-ada-002"

    index_name = 'edge-user'

    # initialize connection to pinecone
    pinecone.init(
        api_key=st.secrets['pinecone_api_key'],  # app.pinecone.io (console)
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
    res = index.query(xq, top_k=4, include_metadata=True,
                      namespace='with-urls')

    # contexts = [
    #     f"##Link:\n{item['metadata']['url']}\n##Body:\n{item['metadata']['text']}" for item in res['matches']]
    # augmented_query = 'Context:\n' + \
    #     "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+'Question:\n'+query

    # primer = f"""You are a customer support agent working for The Edge. You are clear and concise with your answers. If the question is vague, ask for more information. Answer the following question using the context provided. Provite citations to your sources. Respond using markdown."""

    files_string = ""

    for i in range(len(res.matches)):
        result = res.matches[i]
        file_chunk_id = result.id
        score = result.score
        url_link = result.metadata["url"]
        url_text = result.metadata['text']
        file_string = f"###\n\"{url_link}\"\n{url_text}\n"

        files_string += file_string

    messagess = [
        {
            "role": "system",
            "content": f"Given a question, try to answer it using the content of the links below, and if you cannot answer, or find "
            f"a relevant link, just output \"I couldn't find the answer to that question.\".\n\n"
            f"If the answer is not contained in the links or if there are no links, respond with \"I couldn't find the answer "
            f".\" If the question is not actually a question, respond with \"That's not a valid question.\"\n\n"
            f"In the cases where you can find the answer, first give the answer. Then explain how you found the answer from the source or sources, "
            f"and use the exact links of the source info you mention. Do not make up the names of any other links other than those mentioned "
            f"in the links context. Give the answer in markdown format."
            f"Use the following format:\n\nQuestion: <question>\n\nLinks:\n<###\n\"link 1\"\nlink text>\n<###\n\"link 2\"\nlink text>...\n\n"
            f"Answer: <answer or \"I couldn't find the answer to that question\" or \"That's not a valid question.\">\n\n"
            f"Question: {query}\n\n"
            f"Links:\n{files_string}\n"
            f"Answer:"
        },
    ]

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.2,
        messages=messagess,
        stream=True
    )
    # messages=[
    #     {"role": "user", "content": primer},
    #     {"role": "user", "content": augmented_query}
    # ],

    container = st.empty()
    messages: list[str] = []
    for chunk in res:
        chunk_message = chunk['choices'][0]['delta'].get('content', '')
        messages.append(chunk_message)
        container.markdown("".join(messages))

    # st.markdown(res['choices'][0]['message']['content'])
