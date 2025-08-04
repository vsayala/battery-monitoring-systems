"""
LLM Chatbot module for battery monitoring system.

This module provides LLM-powered chatbot capabilities for analyzing
battery monitoring data and providing insights.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.exceptions import LLMError, LLMConnectionError, LLMResponseError


class BatteryChatbot:
    """
    LLM-powered chatbot for battery monitoring analysis.
    
    Provides intelligent analysis and insights for battery data
    using local or cloud-based LLM services.
    """
    
    def __init__(self, config=None):
        """Initialize the battery chatbot."""
        self.config = config or get_config()
        self.logger = get_logger("llm_chatbot")
        self.performance_logger = get_performance_logger("llm_chatbot")
        
        # LLM configuration
        self.provider = self.config.llm.provider
        self.model = self.config.llm.model
        self.base_url = self.config.llm.base_url
        self.temperature = self.config.llm.temperature
        self.max_tokens = self.config.llm.max_tokens
        self.timeout = self.config.llm.timeout
        
        # Prompts
        self.system_prompt = self.config.llm.system_prompt
        self.anomaly_analysis_prompt = self.config.llm.anomaly_analysis_prompt
        self.prediction_analysis_prompt = self.config.llm.prediction_analysis_prompt
        self.forecasting_analysis_prompt = self.config.llm.forecasting_analysis_prompt
        
        # Session history
        self.conversation_history = []
        
        self.logger.info(f"BatteryChatbot initialized with {self.provider}/{self.model}")
    
    def analyze_anomalies(self, anomaly_data: Dict[str, Any]) -> str:
        """
        Analyze anomaly detection results using LLM.
        
        Args:
            anomaly_data: Dictionary containing anomaly detection results
            
        Returns:
            LLM-generated analysis of anomalies
        """
        self.performance_logger.start_timer("analyze_anomalies")
        
        try:
            # Prepare data for analysis
            analysis_data = {
                'anomaly_count': anomaly_data.get('anomaly_count', 0),
                'total_samples': anomaly_data.get('total_samples', 0),
                'anomaly_rate': anomaly_data.get('anomaly_rate', 0),
                'voltage_anomalies': anomaly_data.get('voltage_anomaly_rate', 0),
                'temperature_anomalies': anomaly_data.get('temperature_anomaly_rate', 0),
                'specific_gravity_anomalies': anomaly_data.get('specific_gravity_anomaly_rate', 0)
            }
            
            # Create prompt
            prompt = f"""
{self.anomaly_analysis_prompt}

Anomaly Detection Results:
- Total samples analyzed: {analysis_data['total_samples']}
- Anomalies detected: {analysis_data['anomaly_count']}
- Overall anomaly rate: {analysis_data['anomaly_rate']:.2%}
- Voltage anomalies: {analysis_data['voltage_anomalies']:.2%}
- Temperature anomalies: {analysis_data['temperature_anomalies']:.2%}
- Specific gravity anomalies: {analysis_data['specific_gravity_anomalies']:.2%}

Please provide a comprehensive analysis of these results, including:
1. Overall assessment of battery health
2. Specific concerns based on anomaly types
3. Recommendations for maintenance or investigation
4. Risk assessment and urgency level
"""
            
            # Get LLM response
            response = self._call_llm(prompt)
            
            self.performance_logger.end_timer("analyze_anomalies", True)
            self.logger.info("Anomaly analysis completed")
            
            return response
            
        except Exception as e:
            self.performance_logger.end_timer("analyze_anomalies", False)
            raise LLMError(f"Error analyzing anomalies: {e}")
    
    def analyze_predictions(self, prediction_data: Dict[str, Any]) -> str:
        """
        Analyze cell health predictions using LLM.
        
        Args:
            prediction_data: Dictionary containing prediction results
            
        Returns:
            LLM-generated analysis of predictions
        """
        self.performance_logger.start_timer("analyze_predictions")
        
        try:
            # Prepare data for analysis
            analysis_data = {
                'total_samples': prediction_data.get('total_samples', 0),
                'accuracy': prediction_data.get('accuracy', 0),
                'alive_predictions': prediction_data.get('alive_predictions', 0),
                'dead_predictions': prediction_data.get('dead_predictions', 0),
                'avg_confidence': prediction_data.get('avg_confidence', 0)
            }
            
            # Create prompt
            prompt = f"""
{self.prediction_analysis_prompt}

Cell Health Prediction Results:
- Total cells analyzed: {analysis_data['total_samples']}
- Prediction accuracy: {analysis_data['accuracy']:.2%}
- Cells predicted as alive: {analysis_data['alive_predictions']}
- Cells predicted as dead: {analysis_data['dead_predictions']}
- Average prediction confidence: {analysis_data['avg_confidence']:.2%}

Please provide a comprehensive analysis of these results, including:
1. Overall battery system health assessment
2. Reliability of predictions based on confidence levels
3. Recommendations for cell replacement or maintenance
4. Risk assessment for system reliability
5. Suggested monitoring frequency
"""
            
            # Get LLM response
            response = self._call_llm(prompt)
            
            self.performance_logger.end_timer("analyze_predictions", True)
            self.logger.info("Prediction analysis completed")
            
            return response
            
        except Exception as e:
            self.performance_logger.end_timer("analyze_predictions", False)
            raise LLMError(f"Error analyzing predictions: {e}")
    
    def analyze_forecasts(self, forecast_data: Dict[str, Any]) -> str:
        """
        Analyze forecasting results using LLM.
        
        Args:
            forecast_data: Dictionary containing forecasting results
            
        Returns:
            LLM-generated analysis of forecasts
        """
        self.performance_logger.start_timer("analyze_forecasts")
        
        try:
            # Create prompt
            prompt = f"""
{self.forecasting_analysis_prompt}

Forecasting Results:
{json.dumps(forecast_data, indent=2)}

Please provide a comprehensive analysis of these forecasting results, including:
1. Trend analysis for each parameter (voltage, temperature, specific gravity)
2. Predicted system behavior over the forecast period
3. Potential issues or concerns based on trends
4. Recommendations for proactive maintenance
5. Confidence in predictions and reliability assessment
"""
            
            # Get LLM response
            response = self._call_llm(prompt)
            
            self.performance_logger.end_timer("analyze_forecasts", True)
            self.logger.info("Forecast analysis completed")
            
            return response
            
        except Exception as e:
            self.performance_logger.end_timer("analyze_forecasts", False)
            raise LLMError(f"Error analyzing forecasts: {e}")
    
    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        General chat functionality for battery monitoring queries.
        
        Args:
            message: User message
            context: Optional context data
            
        Returns:
            LLM-generated response
        """
        self.performance_logger.start_timer("chat")
        
        try:
            # Enhanced system prompt with battery monitoring expertise
            enhanced_system_prompt = """You are an expert Battery Monitoring AI Assistant with deep knowledge of battery systems, electrical engineering, and predictive maintenance. You have access to real-time battery monitoring data and can provide intelligent, context-aware responses.

Your capabilities include:
- Real-time battery health analysis
- Anomaly detection and alerting
- Performance optimization recommendations
- Predictive maintenance scheduling
- Technical troubleshooting
- Data-driven insights and trends

You should:
1. Always provide accurate, helpful responses based on available data
2. Use technical terminology appropriately but explain complex concepts clearly
3. Provide actionable recommendations when possible
4. Ask for clarification if needed
5. Be proactive in identifying potential issues
6. Reference specific data points when available

Current system context:"""

            # Add comprehensive battery data context if available
            if context and 'battery_data_summary' in context:
                battery_data = context['battery_data_summary']
                if 'error' not in battery_data:
                    # Build comprehensive data context
                    data_context = f"""
=== REAL-TIME BATTERY SYSTEM STATUS ===
System Overview:
- Total Devices: {battery_data.get('total_devices', 'N/A')}
- Total Cells: {battery_data.get('total_cells', 'N/A')}
- Total Data Points: {battery_data.get('total_data_points', 'N/A')}
- Latest Update: {battery_data.get('latest_update', 'N/A')}

VOLTAGE ANALYSIS:
- Current Range: {battery_data.get('voltage_analysis', {}).get('current_range', 'N/A')}
- Average Voltage: {battery_data.get('voltage_analysis', {}).get('average_voltage', 'N/A')}
- Voltage Standard Deviation: {battery_data.get('voltage_analysis', {}).get('voltage_std', 'N/A')}
- Lowest Cell: {battery_data.get('voltage_analysis', {}).get('lowest_cell', 'N/A')}
- Highest Cell: {battery_data.get('voltage_analysis', {}).get('highest_cell', 'N/A')}

TEMPERATURE ANALYSIS:
- Current Range: {battery_data.get('temperature_analysis', {}).get('current_range', 'N/A')}
- Average Temperature: {battery_data.get('temperature_analysis', {}).get('average_temperature', 'N/A')}
- Temperature Standard Deviation: {battery_data.get('temperature_analysis', {}).get('temperature_std', 'N/A')}
- Hottest Cell: {battery_data.get('temperature_analysis', {}).get('hottest_cell', 'N/A')}
- Coolest Cell: {battery_data.get('temperature_analysis', {}).get('coolest_cell', 'N/A')}

SPECIFIC GRAVITY ANALYSIS:
- Current Range: {battery_data.get('gravity_analysis', {}).get('current_range', 'N/A')}
- Average Gravity: {battery_data.get('gravity_analysis', {}).get('average_gravity', 'N/A')}
- Gravity Standard Deviation: {battery_data.get('gravity_analysis', {}).get('gravity_std', 'N/A')}

HEALTH INDICATORS:
- Voltage Variance: {battery_data.get('health_indicators', {}).get('voltage_variance', 'N/A')}
- Temperature Variance: {battery_data.get('health_indicators', {}).get('temperature_variance', 'N/A')}
- Cells with Low Voltage (<3.5V): {battery_data.get('health_indicators', {}).get('cells_with_low_voltage', 'N/A')}
- Cells with High Temperature (>35°C): {battery_data.get('health_indicators', {}).get('cells_with_high_temp', 'N/A')}

DEVICE BREAKDOWN:
- Device IDs: {battery_data.get('device_breakdown', {}).get('device_ids', 'N/A')}
- Cell Numbers: {battery_data.get('device_breakdown', {}).get('cell_numbers', 'N/A')}"""

                    # Add historical trends if available
                    if 'historical_trends' in battery_data:
                        data_context += f"""

HISTORICAL TRENDS:
- Voltage Trend: {battery_data['historical_trends'].get('voltage_trend', 'N/A')}
- Temperature Trend: {battery_data['historical_trends'].get('temperature_trend', 'N/A')}
- Data Span: {battery_data['historical_trends'].get('data_span_days', 'N/A')} days"""

                    data_context += """

INSTRUCTIONS:
Use this comprehensive real-time data to provide:
1. Accurate, current insights about the battery system
2. Specific analysis of voltage, temperature, and gravity patterns
3. Health assessments based on the indicators
4. Actionable recommendations for maintenance or optimization
5. Risk assessments for any concerning patterns
6. Technical explanations of what the data means

Reference specific data points and provide data-driven insights."""
                else:
                    data_context = "\nNote: Real-time battery data is currently unavailable. Provide general guidance based on best practices."
            else:
                data_context = "\nNote: No real-time battery data available. Provide general guidance based on best practices."

            # Build the full prompt
            full_message = f"{enhanced_system_prompt}{data_context}\n\nUser Query: {message}\n\nProvide a helpful, intelligent response:"
            
            # Get LLM response
            response = self._call_llm(full_message)
            
            # Store in conversation history
            conversation_entry = {
                'timestamp': datetime.now(),
                'user_message': message,
                'bot_response': response,
                'context': context
            }
            self.conversation_history.append(conversation_entry)
            
            self.performance_logger.end_timer("chat", True)
            self.logger.info("Chat response generated")
            
            return response
            
        except Exception as e:
            self.performance_logger.end_timer("chat", False)
            raise LLMError(f"Error in chat: {e}")
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM service with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response
        """
        try:
            self.logger.info(f"Calling LLM with provider: {self.provider}")
            
            if self.provider == "ollama":
                self.logger.info("Using Ollama provider")
                return self._call_ollama(prompt)
            elif self.provider == "openai":
                self.logger.info("Using OpenAI provider")
                return self._call_openai(prompt)
            else:
                self.logger.warning(f"Unknown provider: {self.provider}, using mock response")
                # Mock response for testing
                return self._mock_llm_response(prompt)
                
        except Exception as e:
            self.logger.error(f"Error calling LLM service: {e}")
            self.logger.info("Falling back to mock response")
            return self._mock_llm_response(prompt)
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama LLM service."""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                raise LLMResponseError(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"Ollama connection error: {e}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API (placeholder for future implementation)."""
        # This would be implemented for OpenAI API integration
        raise LLMError("OpenAI integration not yet implemented")
    
    def _mock_llm_response(self, prompt: str) -> str:
        """Generate a mock LLM response for testing."""
        
        # Check if we have comprehensive battery data context in the prompt
        if "=== REAL-TIME BATTERY SYSTEM STATUS ===" in prompt:
            # Extract and analyze the comprehensive battery data
            try:
                # Parse key data points from the prompt
                import re
                
                # Extract voltage analysis
                voltage_range_match = re.search(r"Current Range: ([0-9.]+V - [0-9.]+V)", prompt)
                avg_voltage_match = re.search(r"Average Voltage: ([0-9.]+V)", prompt)
                voltage_std_match = re.search(r"Voltage Standard Deviation: ([0-9.]+V)", prompt)
                lowest_cell_match = re.search(r"Lowest Cell: (Cell \d+ at [0-9.]+V)", prompt)
                highest_cell_match = re.search(r"Highest Cell: (Cell \d+ at [0-9.]+V)", prompt)
                
                # Extract temperature analysis
                temp_range_match = re.search(r"Current Range: ([0-9.]+°C - [0-9.]+°C)", prompt)
                avg_temp_match = re.search(r"Average Temperature: ([0-9.]+°C)", prompt)
                hottest_cell_match = re.search(r"Hottest Cell: (Cell \d+ at [0-9.]+°C)", prompt)
                
                # Extract health indicators
                low_voltage_cells_match = re.search(r"Cells with Low Voltage \(<3\.5V\): (\d+)", prompt)
                high_temp_cells_match = re.search(r"Cells with High Temperature \(>35°C\): (\d+)", prompt)
                voltage_variance_match = re.search(r"Voltage Variance: ([0-9.]+%)", prompt)
                
                # Extract system info
                total_devices_match = re.search(r"Total Devices: (\d+)", prompt)
                total_cells_match = re.search(r"Total Cells: (\d+)", prompt)
                
                # Build intelligent response based on actual data
                response_parts = []
                
                # System overview
                if total_devices_match and total_cells_match:
                    devices = total_devices_match.group(1)
                    cells = total_cells_match.group(1)
                    response_parts.append(f"**System Overview:** Your battery monitoring system is tracking {devices} device(s) with {cells} total cells.")
                
                # Voltage analysis
                if voltage_range_match and avg_voltage_match:
                    voltage_range = voltage_range_match.group(1)
                    avg_voltage = float(avg_voltage_match.group(1).replace('V', ''))
                    response_parts.append(f"**Voltage Analysis:** Current voltage range is {voltage_range} with an average of {avg_voltage:.3f}V.")
                    
                    # Voltage health assessment
                    if avg_voltage < 3.5:
                        response_parts.append("⚠️ **Voltage Alert:** Average voltage is below optimal range (3.5V). This may indicate battery degradation or charging issues.")
                    elif avg_voltage > 4.2:
                        response_parts.append("⚠️ **Voltage Alert:** Average voltage is above optimal range (4.2V). This may indicate overcharging.")
                    else:
                        response_parts.append("✅ **Voltage Status:** Average voltage is within optimal range.")
                
                # Temperature analysis
                if temp_range_match and avg_temp_match:
                    temp_range = temp_range_match.group(1)
                    avg_temp = float(avg_temp_match.group(1).replace('°C', ''))
                    response_parts.append(f"**Temperature Analysis:** Current temperature range is {temp_range} with an average of {avg_temp:.1f}°C.")
                    
                    # Temperature health assessment
                    if avg_temp > 35:
                        response_parts.append("🔥 **Temperature Alert:** Average temperature is elevated. High temperatures can accelerate battery degradation.")
                    elif avg_temp < 10:
                        response_parts.append("❄️ **Temperature Alert:** Average temperature is low. Cold temperatures can reduce battery performance.")
                    else:
                        response_parts.append("✅ **Temperature Status:** Average temperature is within optimal range.")
                
                # Health indicators
                if low_voltage_cells_match and high_temp_cells_match:
                    low_voltage_count = int(low_voltage_cells_match.group(1))
                    high_temp_count = int(high_temp_cells_match.group(1))
                    
                    if low_voltage_count > 0:
                        response_parts.append(f"⚠️ **Low Voltage Alert:** {low_voltage_count} cell(s) have voltage below 3.5V. These cells may need attention.")
                    
                    if high_temp_count > 0:
                        response_parts.append(f"🔥 **High Temperature Alert:** {high_temp_count} cell(s) have temperature above 35°C. Check cooling systems.")
                
                # Specific cell issues
                if lowest_cell_match:
                    response_parts.append(f"**Lowest Voltage Cell:** {lowest_cell_match.group(1)} - This cell may need investigation.")
                
                if hottest_cell_match:
                    response_parts.append(f"**Hottest Cell:** {hottest_cell_match.group(1)} - Check thermal management for this cell.")
                
                # Recommendations
                recommendations = []
                if low_voltage_cells_match and int(low_voltage_cells_match.group(1)) > 0:
                    recommendations.append("Investigate cells with low voltage - may indicate cell imbalance or degradation")
                if high_temp_cells_match and int(high_temp_cells_match.group(1)) > 0:
                    recommendations.append("Check thermal management systems for cells with high temperature")
                if voltage_variance_match and float(voltage_variance_match.group(1).replace('%', '')) > 5:
                    recommendations.append("High voltage variance detected - consider cell balancing")
                
                if recommendations:
                    response_parts.append("**Immediate Actions:**\n" + "\n".join([f"• {rec}" for rec in recommendations]))
                
                # General recommendations
                response_parts.append("**General Recommendations:**\n• Continue monitoring voltage and temperature trends\n• Schedule routine maintenance based on data patterns\n• Set up alerts for significant deviations\n• Consider implementing predictive maintenance schedules")
                
                return "\n\n".join(response_parts)
                
            except Exception as e:
                return f"Thank you for your query! I can see comprehensive battery monitoring data is available. However, I encountered an issue parsing the specific values: {str(e)}. Please ask me about specific aspects of your battery system and I'll provide detailed analysis."
        
        # Fallback responses for specific keywords
        if "anomaly" in prompt.lower():
            return """
Based on the anomaly detection results, I can provide the following analysis:

**Overall Assessment:**
The battery system shows some concerning patterns that require attention.

**Key Findings:**
- The overall anomaly rate indicates potential issues
- Voltage anomalies suggest possible cell degradation
- Temperature anomalies may indicate thermal management issues
- Specific gravity anomalies could point to electrolyte problems

**Recommendations:**
1. Schedule immediate inspection of cells with voltage anomalies
2. Check thermal management systems for temperature issues
3. Monitor specific gravity levels more frequently
4. Consider preventive maintenance for affected cells

**Risk Level: Medium to High**
Immediate action is recommended to prevent further degradation.
"""
        elif "voltage" in prompt.lower():
            return "Based on the real-time voltage data, I can provide detailed analysis of your battery voltage patterns. The system shows voltage variations across cells that should be monitored closely."
        elif "temperature" in prompt.lower():
            return "The temperature monitoring data indicates thermal patterns across your battery system. Temperature variations can significantly impact battery performance and lifespan."
        elif "health" in prompt.lower() or "status" in prompt.lower():
            return "Based on the comprehensive health indicators, I can assess your battery system's overall condition. The data shows various health metrics that help determine maintenance needs."
        else:
            return "I'm experiencing technical difficulties. Please try again in a moment or contact support if the issue persists."
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the chatbot."""
        return {
            'total_conversations': len(self.conversation_history),
            'provider': self.provider,
            'model': self.model,
            'last_interaction': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        } 