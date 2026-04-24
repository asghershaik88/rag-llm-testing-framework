import asyncio

from ragas.metrics.collections import ContextRecall

from conftest import llm_setup


async def context_recall_async(query, response, context):
    recall=ContextRecall(llm=llm_setup())
    score=await recall.ascore(
        user_input=query,
        retrieved_contexts=context,
        reference=response
    )
    return score

def  context_recall_score(query, response, context):
    return asyncio.run(context_recall_async(query, response, context))

