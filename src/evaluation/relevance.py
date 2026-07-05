"""Relevance evaluator for Hybrid RAG system."""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """Evaluator for answer relevance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize relevance evaluator.
        
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
        """Evaluate answer relevance.
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Relevance evaluation results
        """
        prompt = f"""Evaluate the relevance of the answer to the question.

Question:
{query}

Answer:
{answer}

Please evaluate:
1. Does the answer directly address the question?
2. Is the answer relevant to the question?
3. Are there any irrelevant details?

Respond with a JSON object containing:
- relevance_score: 0.0 to 1.0
- is_relevant: true or false (true if score >= 0.7)
- relevance_notes: brief notes on relevance
- relevant_parts: list of parts of the answer that are relevant"""

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
                'relevance_score': result.get('relevance_score', 0.0),
                'is_relevant': result.get('is_relevant', False),
                'relevance_notes': result.get('relevance_notes', ''),
                'relevant_parts': result.get('relevant_parts', [])
            }
            
        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            return {
                'relevance_score': 0.0,
                'is_relevant': False,
                'relevance_notes': 'Evaluation error',
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
        relevant = 0
        
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
            if result.get('is_relevant', False):
                relevant += 1
        
        return {
            'total_cases': total,
            'relevant_cases': relevant,
            'relevance_rate': relevant / total if total > 0 else 0,
            'results': results
        }
