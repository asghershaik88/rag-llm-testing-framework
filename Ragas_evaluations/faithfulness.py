import asyncio
from ragas.metrics.collections import Faithfulness
from conftest import llm_setup


async def faithfulness_async(query, response, context):
    faith = Faithfulness(llm=llm_setup())

    score = await faith.ascore(
        user_input=query,
        response=response,
        retrieved_contexts=context
    )
    return score

# Sync wrapper

def faithfulness_score(query, response, context):
    return asyncio.run(faithfulness_async(query, response, context))

