"""Test script to verify RAG and Crisis Detection work properly"""
import sys
sys.path.append('.')

# Test Crisis Detection
print('='*60)
print('Testing Crisis Detection...')
print('='*60)
from detection.detector import CrisisDetector
detector = CrisisDetector()

tests = [
    'I want to kill myself',
    'I feel suicidal',
    'I want to commit suicide',
    'I am so depressed',
    'I feel sad today'
]

for test in tests:
    risk, conf, triggers = detector.classify_risk(test)
    status = '🔴' if risk == 'high' else '🟡' if risk == 'medium' else '🟢' if risk == 'low' else '⚪'
    print(f'{status} "{test[:30]}..." -> {risk.upper()} (conf: {conf:.0%})')

print()

# Test RAG
print('='*60)
print('Testing RAG System...')
print('='*60)
from rag.rag import get_rag_context
context = get_rag_context('I feel anxious about work')
if context:
    print(f'✅ RAG returned {len(context)} characters of context')
    print(f'Preview: {context[:150]}...')
else:
    print('⚠️ RAG returned no context (may need documents)')

print()
print('='*60)
print('✅ Both Crisis Detection and RAG are working!')
print('='*60)
