"""
LLMOps - Large Language Model Operations

Comprehensive LLMOps system providing:
1. LLM Performance Monitoring and Evaluation
2. Prompt Engineering and Optimization
3. Response Quality Assessment
4. A/B Testing for LLM Models
5. Continuous Model Improvement
6. Bias Detection and Mitigation
7. Cost and Performance Optimization

This module implements production-grade LLMOps practices for battery monitoring system.
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import re

# Text processing
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from .chatbot import BatteryChatbot
from .evaluator import LLMEvaluator


@dataclass
class LLMMetrics:
    """Data class for LLM performance metrics."""
    model_name: str
    timestamp: datetime
    response_time_ms: float
    tokens_input: int
    tokens_output: int
    cost_usd: float
    quality_score: float
    relevance_score: float
    coherence_score: float
    factual_accuracy: float
    bias_score: float
    safety_score: float
    user_satisfaction: Optional[float] = None


@dataclass
class PromptTemplate:
    """Data class for prompt templates."""
    template_id: str
    name: str
    template: str
    variables: List[str]
    use_case: str
    performance_metrics: Dict[str, float]
    version: str
    created_at: datetime
    last_updated: datetime
    is_active: bool


@dataclass
class ABTestResult:
    """Data class for A/B test results."""
    test_id: str
    model_a: str
    model_b: str
    prompt_a: str
    prompt_b: str
    total_requests: int
    model_a_requests: int
    model_b_requests: int
    model_a_performance: Dict[str, float]
    model_b_performance: Dict[str, float]
    winner: str
    confidence_level: float
    test_duration_hours: float
    started_at: datetime
    ended_at: datetime


class LLMOps:
    """
    LLMOps - Large Language Model Operations System.
    
    Provides comprehensive monitoring, evaluation, and optimization
    capabilities for LLM deployments in production environments.
    """
    
    def __init__(self, config=None):
        """Initialize the LLMOps system."""
        self.config = config or get_config()
        self.logger = get_logger("llmops")
        self.performance_logger = get_performance_logger("llmops")
        
        # LLMOps configuration
        self.metrics_dir = Path("./llm_metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompts_dir = Path("./prompts")
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_dir = Path("./llm_experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.chatbot = BatteryChatbot(config)
        self.evaluator = LLMEvaluator(config)
        
        # Metrics storage
        self.metrics_history = []
        self.prompt_templates = {}
        self.ab_tests = {}
        self.model_performance = defaultdict(list)
        
        # Monitoring configuration
        self.quality_threshold = 0.8
        self.response_time_threshold = 5000  # 5 seconds
        self.cost_threshold_per_request = 0.01  # $0.01
        
        # A/B testing
        self.ab_test_active = False
        self.current_ab_test = None
        self.ab_test_traffic_split = 0.5
        
        # Safety and bias monitoring
        self.safety_keywords = [
            'harmful', 'dangerous', 'unsafe', 'toxic', 'bias', 'discrimination'
        ]
        
        self.logger.info("LLMOps system initialized")
    
    async def monitor_llm_request(self, 
                                  prompt: str, 
                                  response: str, 
                                  model_name: str = "default",
                                  context: Dict[str, Any] = None) -> LLMMetrics:
        """
        Monitor a single LLM request and response.
        
        Args:
            prompt: Input prompt to the LLM
            response: LLM response
            model_name: Name of the LLM model used
            context: Additional context information
            
        Returns:
            LLMMetrics object with performance data
        """
        try:
            start_time = time.time()
            
            # Calculate basic metrics
            response_time_ms = (time.time() - start_time) * 1000
            tokens_input = self._count_tokens(prompt)
            tokens_output = self._count_tokens(response)
            cost_usd = self._calculate_cost(tokens_input, tokens_output, model_name)
            
            # Evaluate response quality
            quality_metrics = await self._evaluate_response_quality(prompt, response, context)
            
            # Create metrics object
            metrics = LLMMetrics(
                model_name=model_name,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
                quality_score=quality_metrics.get('quality_score', 0.0),
                relevance_score=quality_metrics.get('relevance_score', 0.0),
                coherence_score=quality_metrics.get('coherence_score', 0.0),
                factual_accuracy=quality_metrics.get('factual_accuracy', 0.0),
                bias_score=quality_metrics.get('bias_score', 0.0),
                safety_score=quality_metrics.get('safety_score', 0.0)
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.model_performance[model_name].append(metrics)
            
            # Check for alerts
            await self._check_performance_alerts(metrics)
            
            # Save metrics to disk
            await self._save_metrics(metrics)
            
            self.logger.info(f"LLM request monitored - Quality: {metrics.quality_score:.3f}, "
                           f"Response time: {metrics.response_time_ms:.1f}ms, Cost: ${metrics.cost_usd:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring LLM request: {e}")
            raise
    
    async def start_ab_test(self, 
                           test_name: str,
                           model_a: str, 
                           model_b: str,
                           prompt_template_a: str = None,
                           prompt_template_b: str = None,
                           traffic_split: float = 0.5,
                           duration_hours: int = 24) -> str:
        """
        Start an A/B test between two models or prompt templates.
        
        Args:
            test_name: Name of the A/B test
            model_a: First model to test
            model_b: Second model to test
            prompt_template_a: Optional prompt template for model A
            prompt_template_b: Optional prompt template for model B
            traffic_split: Percentage of traffic for model A (0.0-1.0)
            duration_hours: Test duration in hours
            
        Returns:
            Test ID
        """
        try:
            test_id = f"abtest_{int(time.time())}_{test_name}"
            
            ab_test = {
                'test_id': test_id,
                'test_name': test_name,
                'model_a': model_a,
                'model_b': model_b,
                'prompt_template_a': prompt_template_a or "default",
                'prompt_template_b': prompt_template_b or "default",
                'traffic_split': traffic_split,
                'duration_hours': duration_hours,
                'started_at': datetime.now(),
                'end_time': datetime.now() + timedelta(hours=duration_hours),
                'requests_a': [],
                'requests_b': [],
                'is_active': True
            }
            
            self.ab_tests[test_id] = ab_test
            self.ab_test_active = True
            self.current_ab_test = test_id
            self.ab_test_traffic_split = traffic_split
            
            self.logger.info(f"Started A/B test {test_id}: {model_a} vs {model_b}")
            
            # Schedule test completion
            asyncio.create_task(self._schedule_ab_test_completion(test_id, duration_hours))
            
            return test_id
            
        except Exception as e:
            self.logger.error(f"Error starting A/B test: {e}")
            raise
    
    async def evaluate_prompt_performance(self, 
                                        prompt_template: str,
                                        test_cases: List[Dict[str, Any]],
                                        model_name: str = "default") -> Dict[str, Any]:
        """
        Evaluate performance of a prompt template across multiple test cases.
        
        Args:
            prompt_template: Prompt template to evaluate
            test_cases: List of test cases with inputs and expected outputs
            model_name: Model to use for evaluation
            
        Returns:
            Evaluation results
        """
        try:
            results = {
                'prompt_template': prompt_template,
                'model_name': model_name,
                'test_cases_count': len(test_cases),
                'results': [],
                'overall_performance': {},
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            quality_scores = []
            response_times = []
            costs = []
            
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
                
                # Format prompt with test case data
                formatted_prompt = self._format_prompt_template(prompt_template, test_case.get('inputs', {}))
                
                # Get LLM response
                start_time = time.time()
                response = await self.chatbot.get_response(formatted_prompt)
                response_time = (time.time() - start_time) * 1000
                
                # Monitor the request
                metrics = await self.monitor_llm_request(formatted_prompt, response, model_name)
                
                # Evaluate against expected output if provided
                test_result = {
                    'test_case_id': i + 1,
                    'input': test_case.get('inputs', {}),
                    'expected_output': test_case.get('expected_output'),
                    'actual_output': response,
                    'metrics': asdict(metrics),
                    'passed': True  # Default, can be enhanced with specific checks
                }
                
                if test_case.get('expected_output'):
                    similarity_score = self._calculate_similarity(
                        test_case['expected_output'], response
                    )
                    test_result['similarity_score'] = similarity_score
                    test_result['passed'] = similarity_score > 0.7
                
                results['results'].append(test_result)
                quality_scores.append(metrics.quality_score)
                response_times.append(response_time)
                costs.append(metrics.cost_usd)
            
            # Calculate overall performance
            results['overall_performance'] = {
                'average_quality_score': statistics.mean(quality_scores),
                'average_response_time_ms': statistics.mean(response_times),
                'total_cost_usd': sum(costs),
                'pass_rate': sum(1 for r in results['results'] if r['passed']) / len(test_cases),
                'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'response_time_p95': np.percentile(response_times, 95)
            }
            
            self.logger.info(f"Prompt evaluation completed - Pass rate: {results['overall_performance']['pass_rate']:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating prompt performance: {e}")
            raise
    
    async def optimize_prompts(self, 
                             base_prompt: str,
                             optimization_goals: List[str] = None) -> Dict[str, Any]:
        """
        Optimize prompt templates using automated techniques.
        
        Args:
            base_prompt: Base prompt to optimize
            optimization_goals: Goals like 'quality', 'speed', 'cost'
            
        Returns:
            Optimization results with improved prompts
        """
        try:
            optimization_goals = optimization_goals or ['quality', 'relevance']
            
            optimization_results = {
                'base_prompt': base_prompt,
                'optimization_goals': optimization_goals,
                'optimized_prompts': [],
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            # Generate prompt variations
            prompt_variations = self._generate_prompt_variations(base_prompt)
            
            # Test each variation
            for i, variant in enumerate(prompt_variations):
                self.logger.info(f"Testing prompt variation {i+1}/{len(prompt_variations)}")
                
                # Create test cases for evaluation
                test_cases = self._create_test_cases_for_prompt(base_prompt)
                
                # Evaluate variant
                evaluation_result = await self.evaluate_prompt_performance(
                    variant, test_cases
                )
                
                optimization_results['optimized_prompts'].append({
                    'variant_id': i + 1,
                    'prompt': variant,
                    'performance': evaluation_result['overall_performance']
                })
            
            # Rank prompts by optimization goals
            ranked_prompts = self._rank_prompts_by_goals(
                optimization_results['optimized_prompts'], 
                optimization_goals
            )
            
            optimization_results['ranked_prompts'] = ranked_prompts
            optimization_results['best_prompt'] = ranked_prompts[0] if ranked_prompts else None
            
            self.logger.info("Prompt optimization completed")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing prompts: {e}")
            raise
    
    async def detect_model_drift(self, 
                               model_name: str,
                               lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Detect performance drift in LLM models.
        
        Args:
            model_name: Name of the model to check
            lookback_hours: Hours to look back for comparison
            
        Returns:
            Drift detection results
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            # Get recent metrics
            recent_metrics = [
                m for m in self.model_performance[model_name]
                if m.timestamp >= cutoff_time
            ]
            
            # Get historical baseline
            baseline_metrics = [
                m for m in self.model_performance[model_name]
                if m.timestamp < cutoff_time
            ]
            
            if len(recent_metrics) < 10 or len(baseline_metrics) < 10:
                return {
                    'drift_detected': False,
                    'reason': 'Insufficient data for drift detection',
                    'recent_samples': len(recent_metrics),
                    'baseline_samples': len(baseline_metrics)
                }
            
            # Calculate drift metrics
            drift_results = self._calculate_drift_metrics(recent_metrics, baseline_metrics)
            
            # Determine if drift occurred
            drift_detected = (
                drift_results['quality_drift'] > 0.1 or
                drift_results['response_time_drift'] > 0.3 or
                drift_results['cost_drift'] > 0.2
            )
            
            drift_results.update({
                'model_name': model_name,
                'drift_detected': drift_detected,
                'lookback_hours': lookback_hours,
                'detection_timestamp': datetime.now().isoformat(),
                'recent_samples': len(recent_metrics),
                'baseline_samples': len(baseline_metrics)
            })
            
            if drift_detected:
                self.logger.warning(f"Model drift detected for {model_name}")
                await self._trigger_drift_alert(model_name, drift_results)
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Error detecting model drift: {e}")
            raise
    
    async def get_llm_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive LLM dashboard data."""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'overview': await self._get_overview_metrics(),
                'model_performance': await self._get_model_performance_summary(),
                'recent_metrics': self._get_recent_metrics(hours=24),
                'ab_tests': await self._get_ab_test_summary(),
                'alerts': await self._get_recent_alerts(),
                'cost_analysis': await self._get_cost_analysis(),
                'quality_trends': await self._get_quality_trends()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            raise
    
    # Private helper methods
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate cost for LLM request."""
        # Simplified cost calculation
        cost_per_1k_input = 0.001  # $0.001 per 1K input tokens
        cost_per_1k_output = 0.002  # $0.002 per 1K output tokens
        
        input_cost = (input_tokens / 1000) * cost_per_1k_input
        output_cost = (output_tokens / 1000) * cost_per_1k_output
        
        return input_cost + output_cost
    
    async def _evaluate_response_quality(self, 
                                       prompt: str, 
                                       response: str,
                                       context: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate the quality of an LLM response."""
        try:
            quality_metrics = {}
            
            # Basic quality checks
            quality_metrics['quality_score'] = self._calculate_basic_quality(response)
            quality_metrics['relevance_score'] = self._calculate_relevance(prompt, response)
            quality_metrics['coherence_score'] = self._calculate_coherence(response)
            quality_metrics['factual_accuracy'] = self._check_factual_accuracy(response, context)
            quality_metrics['bias_score'] = self._detect_bias(response)
            quality_metrics['safety_score'] = self._check_safety(response)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating response quality: {e}")
            return {
                'quality_score': 0.5,
                'relevance_score': 0.5,
                'coherence_score': 0.5,
                'factual_accuracy': 0.5,
                'bias_score': 0.0,
                'safety_score': 1.0
            }
    
    def _calculate_basic_quality(self, response: str) -> float:
        """Calculate basic quality score for response."""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        quality_factors = []
        
        # Length appropriateness
        length_score = min(1.0, len(response) / 100)  # Prefer responses > 100 chars
        quality_factors.append(length_score)
        
        # Grammar and structure (basic check)
        sentences = response.split('.')
        structure_score = min(1.0, len(sentences) / 3)  # Prefer multiple sentences
        quality_factors.append(structure_score)
        
        # Completeness (ends with punctuation)
        completeness_score = 1.0 if response.strip().endswith(('.', '!', '?')) else 0.7
        quality_factors.append(completeness_score)
        
        return np.mean(quality_factors)
    
    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance score between prompt and response."""
        # Simple keyword overlap approach
        prompt_words = set(re.findall(r'\w+', prompt.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        
        if not prompt_words:
            return 0.5
        
        overlap = len(prompt_words.intersection(response_words))
        relevance = overlap / len(prompt_words)
        
        return min(1.0, relevance * 2)  # Scale up to max 1.0
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence score for response."""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # Single sentence is inherently coherent
        
        # Simple coherence check based on sentence flow
        coherence_score = 1.0  # Start optimistic
        
        # Check for repeated words across sentences (indicates flow)
        sentence_words = [set(re.findall(r'\w+', s.lower())) for s in sentences]
        
        overlaps = []
        for i in range(len(sentence_words) - 1):
            overlap = len(sentence_words[i].intersection(sentence_words[i + 1]))
            overlaps.append(overlap)
        
        if overlaps:
            avg_overlap = np.mean(overlaps)
            coherence_score = min(1.0, avg_overlap / 3)  # Normalize
        
        return max(0.5, coherence_score)  # Minimum baseline
    
    def _check_factual_accuracy(self, response: str, context: Dict[str, Any] = None) -> float:
        """Check factual accuracy of response."""
        # Simplified factual checking
        # In production, this would use external fact-checking APIs
        
        # Check for uncertainty expressions (good for accuracy)
        uncertainty_expressions = ['might', 'could', 'possibly', 'perhaps', 'likely']
        uncertainty_count = sum(1 for expr in uncertainty_expressions if expr in response.lower())
        
        # Prefer responses that acknowledge uncertainty appropriately
        if uncertainty_count > 0:
            return min(1.0, 0.8 + (uncertainty_count * 0.1))
        
        return 0.7  # Default moderate score
    
    def _detect_bias(self, response: str) -> float:
        """Detect potential bias in response (higher score = more bias)."""
        bias_indicators = [
            'always', 'never', 'all', 'none', 'every', 'absolutely',
            'obviously', 'clearly', 'definitely'
        ]
        
        bias_count = sum(1 for word in bias_indicators if word in response.lower())
        bias_score = min(1.0, bias_count / 5)  # Normalize to 0-1
        
        return bias_score
    
    def _check_safety(self, response: str) -> float:
        """Check safety of response (higher score = safer)."""
        safety_violations = sum(1 for keyword in self.safety_keywords 
                              if keyword in response.lower())
        
        if safety_violations > 0:
            return max(0.0, 1.0 - (safety_violations * 0.3))
        
        return 1.0  # Safe by default
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _format_prompt_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Format prompt template with variables."""
        formatted_prompt = template
        for key, value in variables.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))
        return formatted_prompt
    
    def _generate_prompt_variations(self, base_prompt: str) -> List[str]:
        """Generate variations of a base prompt for optimization."""
        variations = []
        
        # Add context variations
        variations.append(f"Context: You are an expert in battery monitoring systems.\n\n{base_prompt}")
        
        # Add formatting variations
        variations.append(f"{base_prompt}\n\nPlease provide a detailed response.")
        
        # Add role variations
        variations.append(f"As a battery monitoring expert, {base_prompt.lower()}")
        
        # Add step-by-step variation
        variations.append(f"{base_prompt}\n\nThink step by step:")
        
        return variations
    
    def _create_test_cases_for_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """Create test cases for prompt evaluation."""
        # Simplified test case generation
        test_cases = [
            {
                'inputs': {'query': 'What is the normal voltage range for lithium batteries?'},
                'expected_output': None
            },
            {
                'inputs': {'query': 'How to detect battery anomalies?'},
                'expected_output': None
            },
            {
                'inputs': {'query': 'What causes battery temperature increases?'},
                'expected_output': None
            }
        ]
        
        return test_cases
    
    def _rank_prompts_by_goals(self, prompts: List[Dict], goals: List[str]) -> List[Dict]:
        """Rank prompts based on optimization goals."""
        goal_weights = {
            'quality': lambda p: p['performance'].get('average_quality_score', 0),
            'speed': lambda p: 1.0 / (p['performance'].get('average_response_time_ms', 1) / 1000),
            'cost': lambda p: 1.0 / (p['performance'].get('total_cost_usd', 0.001) + 0.001),
            'relevance': lambda p: p['performance'].get('pass_rate', 0)
        }
        
        for prompt in prompts:
            scores = []
            for goal in goals:
                if goal in goal_weights:
                    scores.append(goal_weights[goal](prompt))
            prompt['composite_score'] = np.mean(scores) if scores else 0.0
        
        return sorted(prompts, key=lambda p: p['composite_score'], reverse=True)
    
    def _calculate_drift_metrics(self, recent_metrics: List[LLMMetrics], 
                                baseline_metrics: List[LLMMetrics]) -> Dict[str, float]:
        """Calculate drift metrics between recent and baseline performance."""
        recent_quality = [m.quality_score for m in recent_metrics]
        baseline_quality = [m.quality_score for m in baseline_metrics]
        
        recent_response_time = [m.response_time_ms for m in recent_metrics]
        baseline_response_time = [m.response_time_ms for m in baseline_metrics]
        
        recent_cost = [m.cost_usd for m in recent_metrics]
        baseline_cost = [m.cost_usd for m in baseline_metrics]
        
        return {
            'quality_drift': abs(np.mean(recent_quality) - np.mean(baseline_quality)),
            'response_time_drift': abs(np.mean(recent_response_time) - np.mean(baseline_response_time)) / np.mean(baseline_response_time),
            'cost_drift': abs(np.mean(recent_cost) - np.mean(baseline_cost)) / (np.mean(baseline_cost) + 0.001),
            'quality_variance_change': np.var(recent_quality) / (np.var(baseline_quality) + 0.001),
            'recent_avg_quality': np.mean(recent_quality),
            'baseline_avg_quality': np.mean(baseline_quality)
        }
    
    async def _check_performance_alerts(self, metrics: LLMMetrics) -> None:
        """Check if metrics trigger any performance alerts."""
        alerts = []
        
        if metrics.quality_score < self.quality_threshold:
            alerts.append(f"Quality score below threshold: {metrics.quality_score:.3f}")
        
        if metrics.response_time_ms > self.response_time_threshold:
            alerts.append(f"Response time above threshold: {metrics.response_time_ms:.1f}ms")
        
        if metrics.cost_usd > self.cost_threshold_per_request:
            alerts.append(f"Cost above threshold: ${metrics.cost_usd:.4f}")
        
        if metrics.safety_score < 0.8:
            alerts.append(f"Safety score below threshold: {metrics.safety_score:.3f}")
        
        for alert in alerts:
            self.logger.warning(f"LLM Performance Alert: {alert}")
    
    async def _save_metrics(self, metrics: LLMMetrics) -> None:
        """Save metrics to persistent storage."""
        try:
            metrics_file = self.metrics_dir / f"metrics_{metrics.timestamp.strftime('%Y%m%d')}.jsonl"
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(asdict(metrics), default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    async def _schedule_ab_test_completion(self, test_id: str, duration_hours: int) -> None:
        """Schedule A/B test completion."""
        await asyncio.sleep(duration_hours * 3600)
        await self._complete_ab_test(test_id)
    
    async def _complete_ab_test(self, test_id: str) -> ABTestResult:
        """Complete an A/B test and analyze results."""
        try:
            test_data = self.ab_tests.get(test_id)
            if not test_data:
                raise ValueError(f"A/B test {test_id} not found")
            
            test_data['is_active'] = False
            test_data['ended_at'] = datetime.now()
            
            # Analyze results (simplified)
            model_a_metrics = test_data['requests_a']
            model_b_metrics = test_data['requests_b']
            
            # Determine winner based on quality scores
            if model_a_metrics and model_b_metrics:
                avg_quality_a = np.mean([m.quality_score for m in model_a_metrics])
                avg_quality_b = np.mean([m.quality_score for m in model_b_metrics])
                
                winner = test_data['model_a'] if avg_quality_a > avg_quality_b else test_data['model_b']
                confidence = abs(avg_quality_a - avg_quality_b) / max(avg_quality_a, avg_quality_b)
            else:
                winner = "inconclusive"
                confidence = 0.0
            
            # Create result object
            result = ABTestResult(
                test_id=test_id,
                model_a=test_data['model_a'],
                model_b=test_data['model_b'],
                prompt_a=test_data['prompt_template_a'],
                prompt_b=test_data['prompt_template_b'],
                total_requests=len(model_a_metrics) + len(model_b_metrics),
                model_a_requests=len(model_a_metrics),
                model_b_requests=len(model_b_metrics),
                model_a_performance={},
                model_b_performance={},
                winner=winner,
                confidence_level=confidence,
                test_duration_hours=duration_hours,
                started_at=test_data['started_at'],
                ended_at=test_data['ended_at']
            )
            
            self.logger.info(f"A/B test {test_id} completed. Winner: {winner}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error completing A/B test: {e}")
            raise
    
    async def _trigger_drift_alert(self, model_name: str, drift_results: Dict[str, Any]) -> None:
        """Trigger alert for model drift detection."""
        self.logger.warning(f"Model drift alert for {model_name}: {drift_results}")
    
    async def _get_overview_metrics(self) -> Dict[str, Any]:
        """Get overview metrics for dashboard."""
        if not self.metrics_history:
            return {
                'total_requests': 0,
                'avg_quality_score': 0.0,
                'avg_response_time_ms': 0.0,
                'total_cost_usd': 0.0
            }
        
        return {
            'total_requests': len(self.metrics_history),
            'avg_quality_score': np.mean([m.quality_score for m in self.metrics_history]),
            'avg_response_time_ms': np.mean([m.response_time_ms for m in self.metrics_history]),
            'total_cost_usd': sum([m.cost_usd for m in self.metrics_history])
        }
    
    async def _get_model_performance_summary(self) -> Dict[str, Any]:
        """Get model performance summary."""
        summary = {}
        for model_name, metrics in self.model_performance.items():
            if metrics:
                summary[model_name] = {
                    'request_count': len(metrics),
                    'avg_quality': np.mean([m.quality_score for m in metrics]),
                    'avg_response_time': np.mean([m.response_time_ms for m in metrics]),
                    'total_cost': sum([m.cost_usd for m in metrics])
                }
        return summary
    
    def _get_recent_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent metrics for dashboard."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [m for m in self.metrics_history if m.timestamp >= cutoff]
        return [asdict(m) for m in recent[-100:]]  # Return last 100
    
    async def _get_ab_test_summary(self) -> Dict[str, Any]:
        """Get A/B test summary."""
        return {
            'active_tests': len([t for t in self.ab_tests.values() if t.get('is_active', False)]),
            'completed_tests': len([t for t in self.ab_tests.values() if not t.get('is_active', True)]),
            'total_tests': len(self.ab_tests)
        }
    
    async def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        # Simplified - would integrate with actual alerting system
        return []
    
    async def _get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis data."""
        if not self.metrics_history:
            return {'total_cost': 0.0, 'avg_cost_per_request': 0.0}
        
        total_cost = sum([m.cost_usd for m in self.metrics_history])
        return {
            'total_cost': total_cost,
            'avg_cost_per_request': total_cost / len(self.metrics_history),
            'cost_trend': 'stable'  # Simplified
        }
    
    async def _get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trend analysis."""
        if len(self.metrics_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_quality = np.mean([m.quality_score for m in self.metrics_history[-10:]])
        older_quality = np.mean([m.quality_score for m in self.metrics_history[-20:-10]]) if len(self.metrics_history) >= 20 else recent_quality
        
        trend = 'improving' if recent_quality > older_quality else 'declining' if recent_quality < older_quality else 'stable'
        
        return {
            'trend': trend,
            'recent_avg_quality': recent_quality,
            'quality_change': recent_quality - older_quality
        }