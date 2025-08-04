'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MessageCircle, 
  X, 
  Send,
  Bot,
  ArrowLeft,
  Battery,
  Users,
  AlertCircle
} from 'lucide-react'
import Link from 'next/link'

interface ChatMessage {
  id: string
  type: 'user' | 'bot'
  content: string
  timestamp: Date
}

export default function ChatPage() {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isTypingWelcome, setIsTypingWelcome] = useState(true)
  const [welcomeText, setWelcomeText] = useState('')
  const [welcomeIndex, setWelcomeIndex] = useState(0)

  const welcomeMessage = `Hello! I'm your intelligent Battery Monitoring AI Assistant powered by advanced LLM technology. I can help you with:

🔋 **Real-time Analysis**: Monitor voltage, temperature, and current across all battery cells
🚨 **Anomaly Detection**: Identify potential issues before they become problems
📊 **Performance Insights**: Get detailed analytics and predictions
⚡ **Optimization**: Suggest efficiency improvements and maintenance schedules
🔍 **Deep Diagnostics**: Analyze historical data and trends
📈 **Predictive Maintenance**: Forecast when maintenance is needed

I can understand natural language queries and provide intelligent, context-aware responses. Try asking me things like:
• "How is my battery system performing today?"
• "Are there any anomalies I should be concerned about?"
• "What's the health status of cell number 5?"
• "When should I schedule the next maintenance?"

How can I assist you today?`

  // Type out welcome message
  useEffect(() => {
    if (isTypingWelcome && welcomeIndex < welcomeMessage.length) {
      const timer = setTimeout(() => {
        setWelcomeText(welcomeMessage.slice(0, welcomeIndex + 1))
        setWelcomeIndex(welcomeIndex + 1)
      }, 30) // Adjust speed here (lower = faster)

      return () => clearTimeout(timer)
    } else if (welcomeIndex >= welcomeMessage.length) {
      setIsTypingWelcome(false)
      // Add the complete welcome message to chat
      setChatMessages([{
        id: '1',
        type: 'bot',
        content: welcomeMessage,
        timestamp: new Date()
      }])
    }
  }, [welcomeIndex, isTypingWelcome, welcomeMessage])

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!chatInput.trim()) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: chatInput,
      timestamp: new Date()
    }

    setChatMessages(prev => [...prev, userMessage])
    const currentInput = chatInput
    setChatInput('')
    setIsTyping(true)

    try {
      console.log('Sending request to backend:', currentInput)
      // Call the real AI backend
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          context: {
            system: 'battery_monitoring',
            user_query: currentInput,
            timestamp: new Date().toISOString()
          }
        })
      })
      
      console.log('Response status:', response.status)
      console.log('Response headers:', response.headers)

      if (response.ok) {
        const data = await response.json()
        console.log('Response data:', data)
        const aiResponse = data.response || "I'm sorry, I couldn't process your request at the moment."
        
        // Type out the AI response
        typeResponse(aiResponse)
      } else {
        // Fallback response if API fails
        const fallbackResponse = "I'm experiencing technical difficulties. Please try again in a moment or contact support if the issue persists."
        typeResponse(fallbackResponse)
      }
    } catch (error) {
      console.error('Error calling chatbot API:', error)
      // Fallback response for network errors
      const errorResponse = "I'm having trouble connecting to my AI brain right now. Please check your connection and try again."
      typeResponse(errorResponse)
    }
  }

  const typeResponse = (response: string) => {
    let index = 0
    const tempMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'bot',
      content: '',
      timestamp: new Date()
    }
    
    setChatMessages(prev => [...prev, tempMessage])
    setIsTyping(false)

    const typeInterval = setInterval(() => {
      if (index < response.length) {
        setChatMessages(prev => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage && lastMessage.type === 'bot') {
            lastMessage.content = response.slice(0, index + 1)
          }
          return newMessages
        })
        index++
      } else {
        clearInterval(typeInterval)
      }
    }, 20) // Adjust typing speed here
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="header">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
                <ArrowLeft className="h-5 w-5 text-white" />
                <span className="text-white font-medium">Back to Dashboard</span>
              </Link>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <Bot className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold gradient-text">Battery AI Assistant</h1>
                  <p className="text-sm text-white/70">Your intelligent monitoring companion</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Chat Interface */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        <div className="chat-modal h-[calc(100vh-200px)] flex flex-col">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {/* Welcome message typing effect */}
            {isTypingWelcome && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="chat-message chat-message-bot">
                  <p className="chat-message-text">
                    {welcomeText}
                    <span className="chat-typing-cursor"></span>
                  </p>
                </div>
              </motion.div>
            )}

            {chatMessages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`chat-message ${message.type === 'user' ? 'chat-message-user' : 'chat-message-bot'}`}>
                  <p className="chat-message-text">
                    {message.content}
                    {message.type === 'bot' && index === chatMessages.length - 1 && 
                     message.content.length < 200 && 
                     <span className="chat-typing-cursor"></span>
                    }
                  </p>
                  <p className="chat-message-timestamp">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </motion.div>
            ))}
            
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="chat-message chat-message-bot">
                  <div className="chat-typing-indicator">
                    <div className="chat-typing-dot"></div>
                    <div className="chat-typing-dot"></div>
                    <div className="chat-typing-dot"></div>
                  </div>
                </div>
              </motion.div>
            )}

            {chatMessages.length === 0 && !isTypingWelcome && !isTyping && (
              <div className="text-center text-white/60 py-12">
                <Bot className="h-16 w-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg mb-2">Ask me anything about your battery system!</p>
                <p className="text-sm">I can help with monitoring, diagnostics, predictions, and optimization.</p>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <form onSubmit={handleChatSubmit} className="p-6 border-t border-white/10 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
            <div className="flex space-x-3">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about battery status, alerts, predictions, or system diagnostics..."
                className="flex-1 premium-input"
                disabled={isTyping}
              />
              <button
                type="submit"
                disabled={!chatInput.trim() || isTyping}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </form>
        </div>

        {/* Quick Actions */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <motion.button
            onClick={() => {
              setChatInput("What's the current battery status?")
            }}
            className="glass-card p-4 text-left hover:bg-white/15 transition-all duration-300"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center space-x-3">
              <Battery className="h-6 w-6 text-blue-400" />
              <div>
                <p className="font-medium text-white">Check Status</p>
                <p className="text-xs text-white/60">Current battery health</p>
              </div>
            </div>
          </motion.button>

          <motion.button
            onClick={() => {
              setChatInput("Are there any active alerts?")
            }}
            className="glass-card p-4 text-left hover:bg-white/15 transition-all duration-300"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-6 w-6 text-red-400" />
              <div>
                <p className="font-medium text-white">View Alerts</p>
                <p className="text-xs text-white/60">Active warnings</p>
              </div>
            </div>
          </motion.button>

          <motion.button
            onClick={() => {
              setChatInput("Generate a performance report")
            }}
            className="glass-card p-4 text-left hover:bg-white/15 transition-all duration-300"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center space-x-3">
              <Users className="h-6 w-6 text-green-400" />
              <div>
                <p className="font-medium text-white">Performance</p>
                <p className="text-xs text-white/60">System report</p>
              </div>
            </div>
          </motion.button>
        </div>
      </main>
    </div>
  )
} 