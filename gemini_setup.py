from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()





def get_response_from_gemini_15_flash(context: str, query: str) -> str:
    """ This will return the response from GPT model 4o mini

    Args:
        context (str): contextual text (get from retriever)
        query (str): user query 

    Returns:
        str: response from the LLM
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

    ## final prompt
    final_prompt: PromptTemplate = template.invoke({
        'context': context,
        'query': query
    })

    ## creating object instance
    llm = GoogleGenerativeAI(
        name = 'apcas 1.0.0',
        model = 'gemini-1.5-flash', 
        verbose = False,
        max_tokens = 1000,
        temperature = 0.3
    )

    response: str = llm.invoke(final_prompt)

    return f"<Gemini 1.5 Flash> {response}"




if __name__ == "__main__":

    context = "india have more that 140Cr citizens"
    print(get_response_from_gemini_15_flash(context, query='how many citizens in india'))
    print(get_response_from_gemini_15_flash(context, query='how many states in india'))



