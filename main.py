import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List
import logging
import json
import warnings
import torch

warnings.filterwarnings('ignore')

class ResumeDataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.model_name = "gpt2"
        print(f"Initializing model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, pad_token='<|endoftext|>')
        self.model = pipeline('text-generation', model=self.model_name, tokenizer=self.tokenizer, truncation=True, device=0 if torch.cuda.is_available() else -1)
        self.df = df
        self.dataset_stats = self.compute_dataset_stats()

    def compute_dataset_stats(self) -> Dict:
        stats = {
            'total_resumes': len(self.df),
            'callback_rate': self.df['received_callback'].mean(),
            'industry_callback_rates': {},
            'job_type_callback_rates': {},
            'avg_experience': self.df['years_experience'].mean(),
            'success_factors': self.analyze_success_factors(),
            'skill_importance': self.analyze_skill_importance()
        }

        for industry in self.df['job_industry'].unique():
            industry_df = self.df[self.df['job_industry'] == industry]
            stats['industry_callback_rates'][industry] = {
                'callback_rate': industry_df['received_callback'].mean(),
                'sample_size': len(industry_df),
                'avg_experience': industry_df['years_experience'].mean()
            }

        for job_type in self.df['job_type'].unique():
            job_df = self.df[self.df['job_type'] == job_type]
            stats['job_type_callback_rates'][job_type] = {
                'callback_rate': job_df['received_callback'].mean(),
                'sample_size': len(job_df),
                'avg_experience': job_df['years_experience'].mean()
            }

        return stats

    def analyze_success_factors(self) -> Dict:
        success_factors = {}
        basic_factors = {
            'computer_skills': 'Computer proficiency',
            'special_skills': 'Specialized skills',
            'honors': 'Academic honors',
            'volunteer': 'Volunteer experience',
            'military': 'Military experience',
            'worked_during_school': 'Work during education',
            'employment_holes': 'Employment gaps',
            'resume_quality': 'Resume quality',
            'has_email_address': 'Email provided'
        }

        for factor, description in basic_factors.items():
            if factor in self.df.columns:
                pos_rate = self.df[self.df[factor] == 1]['received_callback'].mean()
                neg_rate = self.df[self.df[factor] == 0]['received_callback'].mean()
                success_factors[description] = {
                    'positive_impact': pos_rate > neg_rate,
                    'impact_strength': abs(pos_rate - neg_rate),
                    'callback_rate_with': pos_rate,
                    'callback_rate_without': neg_rate
                }

        exp_ranges = [(0, 2), (2, 5), (5, 10), (10, float('inf'))]
        for min_exp, max_exp in exp_ranges:
            mask = (self.df['years_experience'] >= min_exp) & (self.df['years_experience'] < max_exp)
            callback_rate = self.df[mask]['received_callback'].mean()
            success_factors[f'Experience_{min_exp}-{max_exp}'] = {
                'callback_rate': callback_rate,
                'sample_size': mask.sum()
            }

        return success_factors

    def analyze_skill_importance(self) -> Dict:
        importance = {}
        for job_type in self.df['job_type'].unique():
            job_df = self.df[self.df['job_type'] == job_type]
            importance[job_type] = {}
            for skill in ['computer_skills', 'special_skills']:
                if skill in job_df.columns:
                    with_skill = job_df[job_df[skill] == 1]['received_callback'].mean()
                    without_skill = job_df[job_df[skill] == 0]['received_callback'].mean()
                    importance[job_type][skill] = with_skill - without_skill
        return importance

    def create_context_aware_prompt(self, row: pd.Series) -> str:
        job_type = row.get('job_type', 'unknown')
        industry = row.get('job_industry', 'unknown')
        experience = str(row['years_experience']) + " years" if row['years_experience'] != -1 else "unknown"

        skills = []
        if row.get('computer_skills', 0) == 1:
            skills.append("computer skills")
        if row.get('special_skills', 0) == 1:
            skills.append("special skills")

        qualifications = []
        if row.get('honors', 0) == 1:
            qualifications.append("academic honors")
        if row.get('military', 0) == 1:
            qualifications.append("military experience")
        if row.get('volunteer', 0) == 1:
            qualifications.append("volunteer experience")
        if row.get('worked_during_school', 0) == 1:
            qualifications.append("work experience during school")

        prompt = f"""As an expert resume analyzer with access to {self.dataset_stats['total_resumes']} resumes, review this candidate:
POSITION: {job_type} in {industry}
MARKET CONTEXT:
* Industry callback rate: {self.dataset_stats['industry_callback_rates'][industry]['callback_rate']:.1%}
* Similar position callback rate: {self.dataset_stats['job_type_callback_rates'][job_type]['callback_rate']:.1%}
* Typical successful experience: {self.dataset_stats['job_type_callback_rates'][job_type]['avg_experience']:.1f} years
CANDIDATE PROFILE:
* Experience: {experience}
* Skills: {', '.join(skills) if skills else 'No specific skills listed'}
* Qualifications: {', '.join(qualifications) if qualifications else 'None'}
* Resume Quality: {row.get('resume_quality', 'unknown')}
* Employment Gaps: {'Yes' if row.get('employment_holes', 0) == 1 else 'No'}
SUCCESS PATTERNS IN THIS FIELD:
* Most impactful qualifications: {', '.join(k for k, v in self.dataset_stats['success_factors'].items() if isinstance(v, dict) and v.get('positive_impact', False))}
* Required skills impact: {self.dataset_stats['skill_importance'][job_type].get('computer_skills', 0):.1%} callback difference
Based on this data's statistics, provide details on:
1. CALLBACK DECISION (Yes/No) with specific reasoning
2. STRENGTHS vs. successful candidates
3. GAPS vs. successful candidates
4. SPECIFIC IMPROVEMENTS based on success patterns

Use your intelligence to come up with the above 4 details for this candidate
"""
        return prompt

    def analyze_resume(self, row: pd.Series) -> Dict:
        try:
            prompt = self.create_context_aware_prompt(row)
            response = self.model(prompt, max_new_tokens=300, num_return_sequences=1, temperature=0.7)
            analysis = self.parse_response(response[0]['generated_text'])
            print('Analysis@@@@@@@@ ', analysis)
            analysis['context'] = {
                'industry_stats': self.dataset_stats['industry_callback_rates'][row['job_industry']],
                'job_type_stats': self.dataset_stats['job_type_callback_rates'][row['job_type']]
            }
            return analysis
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {
                'error': str(e),
                'callback_recommended': False
            }

    def parse_response(self, text: str) -> Dict:
        sections = {
            'callback_decision': '',
            'strengths': [],
            'gaps': [],
            'improvements': []
        }
        try:
            parts = text.split('\n')
            current_section = ''
            for part in parts:
                if 'CALLBACK DECISION' in part:
                    sections['callback_decision'] = part.split(':')[-1].strip()
                elif 'STRENGTHS' in part:
                    current_section = 'strengths'
                elif 'GAPS' in part:
                    current_section = 'gaps'
                elif 'IMPROVEMENTS' in part:
                    current_section = 'improvements'
                elif part.strip() and current_section:
                    sections[current_section].append(part.strip())
            
            callback_recommended = 'yes' in sections['callback_decision'].lower()
            
            return {
                'callback_recommended': callback_recommended,
                'analysis': sections,
                'full_text': text
            }
        except Exception as e:
            print(f"Parsing error: {str(e)}")
            return {
                'callback_recommended': False,
                'analysis': sections,
                'full_text': text,
                'error': str(e)
            }

def main():
    print("Loading and analyzing resume data...")
    df = create_dataframe()
    analyzer = ResumeDataAnalyzer(df)

    print("\nAnalyzing sample resumes...")
    sample_size = 5
    results = []
    for idx, row in df.sample(n=sample_size, random_state=42).iterrows():
        print(f"Processing resume {idx+1}/{sample_size}")
        analysis = analyzer.analyze_resume(row)
        analysis['actual_callback'] = row['received_callback']
        results.append(analysis)

    # Save results
    with open('resume_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to resume_analysis_results.json")

    # Display sample results
    print("\nSample Analysis Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResume {i}:")
        print("=" * 50)
        if 'error' not in result:
            print(f"Recommendation: {'Callback' if result['callback_recommended'] else 'No Callback'}")
            print(f"Actual: {'Callback' if result['actual_callback'] else 'No Callback'}")
            if 'analysis' in result:
                print("\nKey Findings:")
                for section, items in result['analysis'].items():
                    if items:
                        print(f"\n{section.title()}:")
                        for item in items[:3]:  # Show top 3 items
                            print(f"- {item}")
        else:
            print(f"Analysis failed: {result['error']}")

    return results

if __name__ == "__main__":
    results = main()