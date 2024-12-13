import json
from utils.llm import LLM
from langchain_core.prompts import PromptTemplate

def evaluate(writing_sample: str, writing_question: str, criteria_string: str) -> dict:
    # Define the prompt template
    evaluation_prompt = PromptTemplate.from_template(
        f"""
        ### WRITING SAMPLE:
        {writing_sample}

        ### QUESTION:
        {writing_question}

        ### IELTS WRITING TEST INFORMATION:
        The IELTS Writing test evaluates your ability to express yourself in written English and is designed to assess your writing skills in an academic or general context, depending on whether you are taking the Academic or General Training version of the IELTS test. The {writing_sample} needs to address the {writing_question} and failing to do so will result in a decrease in score. If the {writing_sample} does not meet the requirements for word count, the score will be affected.

        **Test Format**:
        - The IELTS Writing test consists of two tasks:
        - **Task 1**: 
            - Academic: Describe visual information, such as graphs, charts, or diagrams.
            - General Training: Write a letter based on a given situation.
        - **Task 2**: Write an essay in response to a question or statement.

        **Word Count**:
        - Task 1: At least 150 words.
        - Task 2: At least 250 words.

        **Skills Assessed**:
        - Task 1: Present and describe visual information logically and coherently (Academic) or convey information in a clear and appropriate tone (General Training).
        - Task 2: Provide well-structured arguments, opinions, and ideas in written form.

        ### EVALUATION CRITERIA:
        {criteria_string}

        ### INSTRUCTION:
        Evaluate the writing sample based on the provided criteria and score it on a scale from 1 to 9. Use the following steps:
        1. **Task Type Context**: Determine whether the task is Task 1 or Task 2 based on the {writing_question} and evaluate accordingly.
        2. **Task Response**: Assess how well the response addresses the question, its relevance, and development of ideas.
        3. **Coherence and Cohesion**: Evaluate the logical flow, organization of ideas, and use of cohesive devices.
        4. **Lexical Resource**: Examine the range, accuracy, and appropriateness of vocabulary used.
        5. **Grammatical Range and Accuracy**: Analyze the sentence structures, grammar, and punctuation for range and accuracy.

        ### Provide a JSON response(NO PREAMBLE):
        - `score`: The overall band score (1-9).
        - `reasoning`: A detailed explanation for the score, referencing specific points from the writing sample and evaluation criteria.
        - `task_type`: State what task type it was.
        """
    )

    llm = LLM().get_groq_llm()

    chain_evaluation = evaluation_prompt | llm

    try:
        # Invoke the LLM and process the response
        res = chain_evaluation.invoke({
            "writing_sample": writing_sample,
            "writing_question": writing_question,
            "criteria_string": criteria_string,
        })
        
        # Extract and parse the content into a dictionary
        if hasattr(res, 'content'):
            response_dict = json.loads(res.content.strip('```json\n').strip())
            return response_dict
        else:
            raise ValueError("LLM response does not contain `content` field.")
    except Exception as e:
        raise RuntimeError(f"Error during LLM invocation: {str(e)}")
