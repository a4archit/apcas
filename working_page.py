import streamlit as st
from apcas import APCAS





def working_page() -> None:
    st.set_page_config(page_title="APCAS | Working", page_icon='🤖')

    # ----------------------------------------------------------- Sidebar --------------------------------------------------------------- #
    st.sidebar.header('Menu', divider=True)
    st.sidebar.write('It is basically a RAG (Retrieval Augmented Generation) based web application, \
                    that is optimized for chatting with any PDF in an efficient way.')
    
    user_seleccted_model_name = st.sidebar.selectbox(
        index = 0,
        key = 'user_llm_option',
        label = 'Choose LLM model', 
        options = ['Gemini 1.5 Flash','GPT 4o Mini']
    )
    map_model_name: dict = {
        'Gemini 1.5 Flash':'gemini-1.5-flash',
        'GPT 4o Mini':'gpt-4o-mini'
    }

    # extracting right model name
    model_name: str = map_model_name[user_seleccted_model_name]
    



    st.sidebar.subheader("Connect with me!")
    st.sidebar.write("[Kaggle](https://www.kaggle.com/architty108)")
    st.sidebar.write("[Github](https://www.github.com/a4archit)")
    st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/a4archit)")

    if st.sidebar.button('Go to Home', type='primary'):
        st.session_state.page = 'home'

    # --------------------------------------------------------- Body ------------------------------------------------------------------- #

    # Initialize chat mode if not set
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = False

    # If not chatting yet, show uploader + button
    if not st.session_state.chat_mode:
        st.header('APCAS', divider=True)

        uploaded_file = st.file_uploader(
            label = "Upload PDF",
            type = 'pdf',
            accept_multiple_files = False,
            help = "You can upload your PDF file here."
        )

        if uploaded_file:
            if st.button(label = 'Chat with this PDF', type='primary'):
                with st.spinner(text="Loading document..."):
                    # Switch to chat mode
                    st.session_state.chat_mode = True
                    # Store the uploaded file for later use if needed
                    st.session_state.uploaded_file = uploaded_file
                    # Clear old messages if any
                    st.session_state.messages = []

                    # You can read it directly
                    bytes_data = uploaded_file.read()

                    # Or save it to a file to get a path
                    with open("user_uploaded_file.pdf", "wb") as f:
                        f.write(bytes_data)

                    # Now you have a path
                    file_path = "user_uploaded_file.pdf"

                with st.spinner(text="Building model..."):
                    # st.write(uploaded_file._file_urls.upload_url)
                    # creating apcas model instance
                    st.session_state.apcas = APCAS(pdf_path=file_path, model=model_name)

    

    # If in chat mode, show chat
    if st.session_state.chat_mode:
        st.header('APCAS Chat')

        # Try another PDF button
        if st.button('📄 Try another PDF'):
            # Reset state to go back to upload mode
            st.session_state.chat_mode = False
            st.session_state.uploaded_file = None
            st.session_state.messages = []

            # Stop here to prevent rendering the rest
            st.rerun()
        

        # Initialize session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input box
        prompt = st.chat_input("Type your message...")

        # When user submits a message
        if prompt:
            # Save user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Getting response..."):
                response = st.session_state.apcas.run(query=prompt, model=model_name)

                # Save bot message
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Display bot message
            with st.chat_message("assistant"):
                st.write(response)









if __name__ == "__main__":
    working_page()









