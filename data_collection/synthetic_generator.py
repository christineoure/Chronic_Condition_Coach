# data_collection/synthetic_generator.py
"""
Synthetic Data Generator for Chronic Conditions
Generates realistic medical content using LLMs when real data is insufficient
"""

import openai  # or your preferred LLM provider
import json
import logging
from typing import List, Dict, Optional
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic medical content using LLMs"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize synthetic data generator
        
        Args:
            api_key: LLM API key (optional - will use mock if not provided)
            model: LLM model to use
        """
        self.model = model
        self.use_real_llm = False
        
        if api_key:
            try:
                openai.api_key = api_key
                # Test the API
                self.use_real_llm = True
                logger.info("Real LLM API enabled")
            except:
                logger.warning("LLM API key invalid, using mock generator")
                self.use_real_llm = False
        else:
            logger.info("No API key provided, using mock generator")
    
    def generate_medical_content(self, topics: List[str], num_documents: int = 20) -> List[Dict]:
        """
        Generate synthetic medical documents
        
        Args:
            topics: List of medical topics
            num_documents: Number of documents to generate
            
        Returns:
            List of generated documents
        """
        documents = []
        
        for i in range(num_documents):
            topic = random.choice(topics)
            doc_type = random.choice(['patient_guide', 'clinical_summary', 'faq', 'research_synopsis'])
            
            if self.use_real_llm:
                document = self._generate_with_llm(topic, doc_type, i)
            else:
                document = self._generate_mock(topic, doc_type, i)
            
            documents.append(document)
            
            # Rate limiting
            if self.use_real_llm:
                time.sleep(0.5)  # Avoid rate limits
        
        return documents
    
    def _generate_with_llm(self, topic: str, doc_type: str, doc_id: int) -> Dict:
        """Generate document using real LLM API"""
        try:
            prompt = self._create_prompt(topic, doc_type)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical writer creating accurate, helpful content for patients and healthcare providers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            return {
                'doc_id': f'synthetic_{doc_id}',
                'topic': topic,
                'type': doc_type,
                'title': self._extract_title(content),
                'content': content,
                'source': 'synthetic_llm',
                'generated_at': datetime.now().isoformat(),
                'metadata': {
                    'model': self.model,
                    'word_count': len(content.split())
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            # Fallback to mock
            return self._generate_mock(topic, doc_type, doc_id)
    
    def _generate_mock(self, topic: str, doc_type: str, doc_id: int) -> Dict:
        """Generate mock document without LLM API"""
        
        # Template-based generation
        templates = {
            'patient_guide': [
                f"## Understanding {topic}\n\n{topic} is a chronic condition that requires careful management. This guide provides practical information for patients.",
                f"## Living with {topic}\n\nManaging {topic} involves lifestyle modifications, medication adherence, and regular monitoring.",
                f"## {topic}: A Patient's Guide\n\nLearn about symptoms, treatments, and self-care strategies for {topic}."
            ],
            'clinical_summary': [
                f"## Clinical Overview: {topic}\n\n{topic} affects millions worldwide. Key considerations include diagnosis criteria, treatment options, and prognosis.",
                f"## {topic} Management Guidelines\n\nEvidence-based approaches to {topic} management including pharmacological and non-pharmacological interventions."
            ],
            'faq': [
                f"## Frequently Asked Questions about {topic}\n\n**Q: What are the main symptoms?**\nA: Common symptoms include...\n\n**Q: How is it diagnosed?**\nA: Diagnosis typically involves...",
                f"## {topic}: Questions and Answers\n\nAddressing common patient concerns about {topic} treatment and management."
            ],
            'research_synopsis': [
                f"## Recent Advances in {topic} Research\n\nNew studies show promising developments in {topic} treatment and understanding.",
                f"## {topic}: Current Research Landscape\n\nSummary of ongoing clinical trials and research directions for {topic}."
            ]
        }
        
        template = random.choice(templates[doc_type])
        
        # Add more detailed content based on topic
        content_mapping = {
            'diabetes': "Focus on blood glucose monitoring, carbohydrate counting, medication management, and complication prevention.",
            'hypertension': "Emphasize regular blood pressure monitoring, medication adherence, sodium restriction, and stress management.",
            'arthritis': "Discuss pain management strategies, joint protection techniques, exercise modifications, and assistive devices.",
            'asthma': "Cover inhaler techniques, trigger avoidance, action plans for exacerbations, and environmental controls.",
            'heart disease': "Include information on cardiac rehabilitation, dietary modifications, medication management, and symptom monitoring."
        }
        
        detailed_content = content_mapping.get(topic.lower(), "Regular follow-up with healthcare providers is essential for optimal management.")
        
        full_content = f"{template}\n\n### Key Management Strategies\n\n{detailed_content}\n\n### Important Considerations\n\n- Always consult with your healthcare provider before making changes to your treatment plan\n- Individual responses to treatments may vary\n- Regular monitoring and follow-up are crucial\n\n*This is educational content and not a substitute for professional medical advice.*"
        
        return {
            'doc_id': f'synthetic_{doc_id}',
            'topic': topic,
            'type': doc_type,
            'title': f"{topic.replace('_', ' ').title()} - {doc_type.replace('_', ' ').title()}",
            'content': full_content,
            'source': 'synthetic_mock',
            'generated_at': datetime.now().isoformat(),
            'metadata': {
                'model': 'mock_generator',
                'word_count': len(full_content.split())
            }
        }
    
    def _create_prompt(self, topic: str, doc_type: str) -> str:
        """Create prompt for LLM generation"""
        prompts = {
            'patient_guide': f"Create a comprehensive patient guide about {topic}. Include information on symptoms, treatment options, lifestyle modifications, and when to seek medical help. Format with clear headings and bullet points where appropriate.",
            'clinical_summary': f"Write a clinical summary about {topic} for healthcare professionals. Include epidemiology, pathophysiology, diagnostic criteria, treatment guidelines, and prognosis.",
            'faq': f"Create a Frequently Asked Questions document about {topic}. Include 8-10 common questions with detailed, evidence-based answers.",
            'research_synopsis': f"Summarize current research and recent advances in {topic}. Include information on ongoing clinical trials, new treatments, and future directions."
        }
        
        return prompts.get(doc_type, f"Write about {topic} from a medical perspective.")
    
    def _extract_title(self, content: str) -> str:
        """Extract title from generated content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
            if line.startswith('## '):
                return line[3:].strip()
        
        # Fallback: use first line or generic title
        first_line = lines[0].strip() if lines else "Medical Document"
        if len(first_line) > 100:
            return first_line[:100] + "..."
        return first_line
    
    def generate_qna_pairs(self, topic: str, num_pairs: int = 10) -> List[Dict]:
        """Generate Q&A pairs for a specific topic"""
        qna_pairs = []
        
        common_questions = {
            'diabetes': [
                "What foods should I avoid with diabetes?",
                "How often should I check my blood sugar?",
                "What are the symptoms of low blood sugar?",
                "Can diabetes be reversed?",
                "What exercises are best for diabetes management?"
            ],
            'hypertension': [
                "What is considered normal blood pressure?",
                "How can I lower my blood pressure naturally?",
                "What are the risks of untreated high blood pressure?",
                "How often should I monitor my blood pressure?",
                "Are there foods that help lower blood pressure?"
            ],
            'general': [
                "What are the common symptoms?",
                "How is this condition diagnosed?",
                "What treatment options are available?",
                "Are there lifestyle changes that can help?",
                "When should I seek emergency care?"
            ]
        }
        
        questions = common_questions.get(topic.lower(), common_questions['general'])
        
        for i in range(min(num_pairs, len(questions))):
            question = questions[i] if i < len(questions) else f"What should I know about {topic}?"
            
            if self.use_real_llm:
                answer = self._generate_answer_with_llm(question, topic)
            else:
                answer = self._generate_mock_answer(question, topic)
            
            qna_pairs.append({
                'question': question,
                'answer': answer,
                'topic': topic,
                'source': 'synthetic_qna'
            })
        
        return qna_pairs
    
    def _generate_answer_with_llm(self, question: str, topic: str) -> str:
        """Generate answer using LLM"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a medical expert answering questions about {topic}."},
                    {"role": "user", "content": question}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._generate_mock_answer(question, topic)
    
    def _generate_mock_answer(self, question: str, topic: str) -> str:
        """Generate mock answer without LLM"""
        answer_templates = [
            f"For {topic}, it's important to note that {question.lower().replace('?', '')} requires consideration of individual factors. Generally, patients should follow their healthcare provider's recommendations.",
            f"When managing {topic}, {question.lower().replace('?', '')} involves a combination of medical treatment and lifestyle modifications. Regular monitoring and follow-up are essential.",
            f"{question} The answer depends on the specific circumstances of the patient with {topic}. Consultation with a healthcare professional is recommended for personalized advice."
        ]
        
        return random.choice(answer_templates)