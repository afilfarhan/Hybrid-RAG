"""Hallucination evaluator for Hybrid RAG system."""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """Evaluator for answer hallucination."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hallucination evaluator.
        
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
        """Evaluate answer for hallucination.
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Hallucination evaluation results
        """
        if not context:
            return {
                'hallucination_score': 1.0,
                'is_hallucinated': True,
                'issues': ['No context provided for comparison']
            }
        
        context_text = "\n".join(doc.get('text', '') for doc in context)
        
        prompt = f"""Detect hallucinations in the answer.

Context:
{context_text}

Question:
{query}

Answer:
{answer}

Please detect:
1. Facts stated in the answer that are not in the context
2. Contradictions with the context
3. Unsupported claims or assumptions
4. Made-up information

Respond with a JSON object containing:
- hallucination_score: 0.0 to 1.0 (0 = no hallucination, 1 = full hallucination)
- is_hallucinated: true or false (true if score >= 0.3)
- hallucination_types: list of hallucination types found
- specific_issues: list of specific issues with line-by-line analysis"""

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
                'hallucination_score': result.get('hallucination_score', 0.0),
                'is_hallucinated': result.get('is_hallucinated', False),
                'hallucination_types': result.get('hallucination_types', []),
                'specific_issues': result.get('specific_issues', [])
            }
            
        except Exception as e:
            logger.error(f"Error evaluating hallucination: {e}")
            return {
                'hallucination_score': 1.0,
                'is_hallucinated': True,
                'hallucination_types': ['evaluation_error'],
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
        hallucinated = 0
        
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
            if result.get('is_hallucinated', False):
                hallucinated += 1
        
        return {
            'total_cases': total,
            'hallucinated_cases': hallucinated,
            'hallucination_rate': hallucinated / total if total > 0 else 0,
            'results': results
        }
