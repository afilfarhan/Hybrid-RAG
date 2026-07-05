"""Sample golden test set data for evaluation."""

test_cases = [
    {
        "query": "What is your return policy?",
        "expected_answer": "We offer a 30-day return policy for all products.",
        "expected_context_sources": ["returns_policy.pdf"]
    },
    {
        "query": "How long does shipping take?",
        "expected_answer": "Standard shipping takes 3-5 business days.",
        "expected_context_sources": ["shipping_info.pdf"]
    },
    {
        "query": "Do you offer international shipping?",
        "expected_answer": "Yes, we ship to over 50 countries worldwide.",
        "expected_context_sources": ["shipping_info.pdf"]
    },
    {
        "query": "What payment methods do you accept?",
        "expected_answer": "We accept credit cards, PayPal, and bank transfers.",
        "expected_context_sources": ["payment_methods.pdf"]
    },
    {
        "query": "Can I track my order?",
        "expected_answer": "Yes, you can track your order using the tracking number sent to your email.",
        "expected_context_sources": ["order_tracking.pdf"]
    }
]
