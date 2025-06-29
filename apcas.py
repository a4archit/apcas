"""
                    APCAS (Any PDF Chatting AI System)
                    =========================================================================================================

                    VERSION = 1.0.0

                    APCAS stands for Any PDF Chatting AI System, it is basically a RAG (Retrieval Augmented Generation) based
                    application, that optimized for chatting with any PDF in an efficint way, still it is not a professional 
                    project So it may have several bugs.


                    Example:
                        ```
                            from apcas import APCAS
                            model = APCAS('path_of_pdf')
                            model.run_on_terminal()
                        ```
                            
"""


VERSION = '1.0.0'




## dependecies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate

from typing import List, Self, Optional
from dotenv import load_dotenv

import os








class APCAS:
    ''' Main class '''


    __name__    = "APCAS (Any PDF Chatting AI System)"
    __model__   = 'Gemini 1.5 Flash'
    __version__ = VERSION # 1.0.0



    def __init__(
            self, 
            pdf_path: Optional[str] = None,
            testing_phase: bool = False
        ) -> Self :

        ''' Constructor '''

        self.testing = testing_phase
        
        ## defining instance variables
        self.model = None
        self.loaded_text: None | List[Document] = None
        self.chunks: None | List = None
        self.embedding_model = None 
        self.vector_store: None | Chroma = None
        self.template = None
        self.retriever: None | Chroma.as_retriever = None
        self.pdf_path: None | str = pdf_path

        try:
            ## loading environment
            load_dotenv()
        except Exception as e:
            print("Something gone wrong with loading environment")


        if self.pdf_path:
            print("Loading all components..... Please wait.....")
            self.setup()

        if self.testing:
            print('APCAS: object created successfully!')

    
    

    def update_pdf_path(self, path: str) -> None:
        ''' this will update the pdf given path '''
        
        ## checking is file exists or not
        if not os.path.exists(path):
            raise FileNotFoundError(f"Given file doesn't exists, provide a right file path.")
        
        ## updating path
        self.pdf_path = path

        if self.testing:
            print('Path updated')



    def load_model(self, is_return: bool = False) -> None | GoogleGenerativeAI :
        ''' this will return the llm after loading '''

        ## creating object instance
        llm = GoogleGenerativeAI(
            name = 'apcas 1.0.0',
            model = 'gemini-1.5-flash', 
            verbose = False,
            max_tokens = 1000,
            temperature = 0.3
        )

        ## updating model
        self.model = llm

        if is_return:
            return llm





    def load_pdf(self, is_return: bool = False) -> None | List :
        ''' this will return the loaded pdf text, using PyPDfLoader '''

        ## cheking for path avialability
        if not self.pdf_path:
            raise ValueError("Before loading PDF must be call model.update_pdf_path() first.")
        
        ## pdf loader instance
        loader = PyPDFLoader(file_path=self.pdf_path, mode='single')
        text = loader.load()[0].page_content ## loading text

        if self.testing:
            print("PDF loaded")

        ## updatng 
        self.loaded_text = text 

        if is_return:
            return text
    



    def split_text(self, large_text: Optional[str] = None, is_return: bool = False) -> List:
        ''' this will return the chunks from provided text '''
        
        ## checking is user provide large text or not
        if not large_text:
            if not self.loaded_text:
                raise ValueError(f"You doen't provide text yet. You have two options to solve this issue: \n1. Provide text as attribute\n2. call obj.pdf_loader() ")
            

            ## if user provide text (pdf path) already, then
            text = self.loaded_text


        if not text:
            ## if user provide text as the attribute of this function
            text = large_text

        if not text:
            raise ValueError("Path of PDF and simple text as attribute of split_text() method is not provided yet. Please provied PDF path or simple text first.")

        ## creating instance of text splitter 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        chunks = text_splitter.split_text(text)

        if self.testing:
            print('Text splits successfully')

        ## updating chunks
        self.chunks = chunks 

        if is_return:
            return chunks
        




    def create_vector_store(self, chunks: Optional[List[str]] = None, is_return: bool = False) :
        ''' In this, embeddings of chunks will generate and store in FAISS (vector store) '''
        if not chunks:
            chunks = self.chunks
            if not chunks:
                raise ValueError("Must be provide `chunks` or PDF path before calling `get_vector_store()`")
            
        ## creating embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        ## updating embedding model
        self.embedding_model = embedding_model

        if self.testing :
            print("Embedding model created.")

        ## creating vector store database
        vector_store = FAISS.from_texts(
            texts = chunks,
            embedding = self.embedding_model
        )

        # ## saving vector store
        # new_dir = os.path.join(os.getcwd(), 'APCAS_FAISS_Database')
        # vector_store.save_local(new_dir)

        ## updating vactor store
        self.vector_store = vector_store


        if self.testing:
            print('Vector Store Created')

        if is_return:
            return vector_store 
        




    def create_retriever(self, vector_store: Optional[FAISS] = None, is_return: bool = False):
        ''' In this function retriever will created and updated '''

        if not vector_store:
            vector_store = self.vector_store
            if not vector_store:
                raise ValueError("Vector store not completed yet, Check attribute and you provide pdf path or not")

        ## creating retriver    
        retriever = vector_store.as_retriever()

        ## updating retriever
        self.retriever = retriever


        if self.testing:
            print('Retriever created')

        if is_return:
            return retriever
        




    def setup(self) -> None:
        """ It will call the all neccessary functions and making sure the model working """
        
        ## loading pdf
        self.load_pdf()

        ## text splitting
        self.split_text()

        ## embedding and vector storing
        self.create_vector_store()

        ## setup retriever
        self.create_retriever()

        ## model loading
        self.load_model()

        ## template loading
        self.load_template()


        print("All setup complete.")







    def load_template(self) -> PromptTemplate :
        """Loading common prompt template

        Returns:
            PromptTemplate: Template for Augmentation 
        """

        ## creating template
        template = PromptTemplate(
            name = 'APCAS common query template',
            template = """
            You are a good assistant.
            Answer the query from the given context ONLY, if no answer will found then say you don't know.

            context:  {context}
            query: {query}
            """,
            input_variables = ['context','query'],
            validate_template = True
        )

        ## updating template
        self.template = template







    def run(self, query: str) -> str:
        """Get response of your single query

        Args:
            query (str): Any question from your side regarding provided PDF

        Returns:
            str: response of the AI system
        """

        ## cheking for valid retriver
        if self.retriever is None:
            raise Exception("Please activate retriver first! \nYou can call `obj.create_retriever()`")
        
        ## getting context
        context = "\n\n".join([doc.page_content for doc in self.retriever.invoke(query)])

        ## final prompt
        final_prompt: PromptTemplate = self.template.invoke({
            'context': context,
            'query': query
        })

        response: str = self.model.invoke(final_prompt)

        return response
    



    def run_on_terminal(self) -> None:
        """ Activate Chat AI feature in your terminal """

        print(f"{'-'*50} APCAS ({self.__version__}) has been activated! Now you can chat with it! {'-'*50}")

        while True:

            query = input("\n\n\033[33m-----(YOU): ").lower().strip()
            if query == "":
                continue
            if query in ["exit","bye","bye bye"]:
                print("\n\n\tBye Bye!\n\n")
                break

            response = self.run(query)

            # displaying ouptut
            print("\n\n\033[32m-----(AI Response)\n\t|")
            for line in response.split('\n'):
                print(f"\t| {line}")
            print("\t|")
    







if __name__ == "__main__":

    path = '/home/archit-elitebook/Downloads/(DT)Data Mining 4th sem Question Bank Solution 2K25 (1).pdf'
    apcas = APCAS(pdf_path=path, testing_phase=False)
    apcas.run_on_terminal()

    # print(apcas.loaded_text)

    # apcas.update_pdf_path('/home/archit-elitebook/Downloads/(DT)Data Mining 4th sem Question Bank Solution 2K25 (1).pdf')

    # apcas.load_pdf()

    # apcas.split_text()

    # # print(type(apcas.chunks))
    # # for i in apcas.chunks[-3:]:
    # #     print(type(i), i,end=f"\n\n{'-'*100}\n")

    # apcas.create_vector_store()

    # apcas.create_retriever()

    # apcas.load_model()

    # apcas.load_template()

    # print(apcas.retriever.invoke("What is data mining", k=3))

    # print(apcas.run("what is knn"))
    # print(apcas.run("what is genai"))





