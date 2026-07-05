"""Faithfulness evaluator."""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:
    """Evaluator for answer faithfulness."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize faithfulness evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'gpt-4o')
        
    async def evaluate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        answer: str
    ) -> Dict[str, Any]:
        """Evaluate answer faithfulness.
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Faithfulness evaluation results
        """
        if not context:
            return {
                'faithfulness_score': 0.0,
                'is_faithful': False,
                'issues': ['No context provided']
            }
        
        context_text = "\n".join(doc.get('text', '') for doc in context)
        
        prompt = f"""Evaluate the faithfulness of the answer based on the context.

Context:
{context_text}

Question:
{query}

Answer:
{answer}

Please evaluate:
1. Does the answer only state what's supported by the context?
2. Are there any unsupported claims?
3. Are there any contradictions with the context?

Respond with a JSON object containing:
- faithfulness_score: 0.0 to 1.0
- is_faithful: true or false (true if score >= 0.7)
- issues: list of any issues found
- supporting_evidence: list of evidence from context that supports the answer"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an evaluation assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'faithfulness_score': result.get('faithfulness_score', 0.0),
                'is_faithful': result.get('is_faithful', False),
                'issues': result.get('issues', []),
                'supporting_evidence': result.get('supporting_evidence', [])
            }
            
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return {
                'faithfulness_score': 0.0,
                'is_faithful': False,
                'issues': ['Evaluation error'],
                'error': str(e)
            }
    
    async def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            Batch evaluation results
        """
        results = []
        total = 0
        faithful = 0
        
        for case in test_cases:
            result = await self.evaluate(
                case.get('query', ''),
                case.get('context', []),
                case.get('answer', '')
            )
            
            results.append({
                'query': case.get('query', ''),
                'result': result
            })
            
            total += 1
            if result.get('is_faithful', False):
                faithful += 1
        
        return {
            'total_cases': total,
            'faithful_cases': faithful,
            'faithfulness_rate': faithful / total if total > 0 else 0,
            'results': results
        }
