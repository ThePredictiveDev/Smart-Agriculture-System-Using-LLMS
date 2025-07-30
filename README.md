# 🌾 Smart Agriculture System Using LLMs

A comprehensive AI-powered agricultural analytics platform that provides real-time insights, predictions, and recommendations for smart farming using Large Language Models (LLMs).

## 🚀 Features

### Core Analytics
- **Real-time Sensor Data Analysis**: Processes temperature, humidity, soil moisture, and NPK values
- **Weather Integration**: Live weather data from wttr.in for Mandi, Himachal Pradesh
- **AI-Powered Predictions**: LLM-based recommendations for crop management
- **Interactive Dashboard**: Modern, responsive UI with real-time visualizations

### Prediction Capabilities
- **Soil Analysis**: Soil type classification, fertilization status, and fertilizer recommendations
- **Crop Management**: Optimal crop recommendations, harvest timing, and selling strategies
- **Threat Detection**: Real-time pest and disease threat assessment
- **Irrigation Guidance**: Smart irrigation recommendations based on soil moisture and weather

### Interactive Features
- **Keyboard Shortcuts**: Press keys (k, l, t, y, m, n, s, d) to simulate different sensor scenarios
- **Multiple Query Modes**: All predictions, soil-specific, crop-specific, or custom prediction sets
- **Enhanced Visualizations**: Trend charts, status badges, and AI recommendation displays
- **Weather Forecasting**: 3-day weather forecast with visual indicators

## 🛠️ Technology Stack

- **Backend**: Python 3.11+
- **AI/ML**: Groq API (DeepSeek R1 Distill Llama 70B)
- **Web Framework**: Gradio
- **Weather Data**: wttr.in API
- **Caching**: requests-cache
- **Data Processing**: JSON Schema validation

## 📋 Prerequisites

- Python 3.11 or higher
- Groq API token
- Internet connection for weather data and LLM API calls

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Smart-Agriculture-System-Using-LLMS
   ```

2. **Install dependencies**
   ```bash
   pip install gradio requests requests-cache jsonschema
   ```

3. **Set up environment variables**
   ```bash
   # Set your Groq API token
   export GROQ_API_TOKEN="your_groq_api_token_here"
   
   # Optional: Set custom Groq API URL (defaults to https://api.groq.com/openai/v1/chat/completions)
   export GROQ_API_URL="https://api.groq.com/openai/v1/chat/completions"
   ```

## 🚀 Usage

### Running the Application

1. **Start the application**
   ```bash
   python dp_smart_agri.py
   ```

2. **Access the interface**
   - The application will launch a local web server
   - Open your browser to the provided URL
   - For public access, the app will also generate a shareable link

### Using the Interface

#### Basic Operation
1. **Select Query Mode**: Choose from available prediction modes
2. **Configure Parameters**: Set specific predictions or advanced prompts if needed
3. **Click "Run"**: Execute the analysis with current sensor and weather data

#### Interactive Features
- **Keyboard Shortcuts**: Press any of these keys to simulate different sensor scenarios:
  - `k`, `l`: Different soil moisture levels (55%, 65%)
  - `t`, `y`: Moderate NPK values with varying moisture
  - `m`, `n`: Low NPK values with high moisture
  - `s`, `d`: High NPK values with moderate moisture

- **Manual Input**: Use the discrete input field at the bottom for custom sensor scenarios

#### Dashboard Tabs
- **Smart Dashboard**: Enhanced visualizations with charts, status badges, and AI recommendations
- **Raw Data**: Detailed sensor data, weather information, and prediction results

## 📊 Prediction Categories

### Soil-Related Predictions
- Soil type classification (Loamy, Clayey, Red)
- Fertilization status (Over, Under, Optimal)
- Fertilizer recommendations (N:P:K ratios)

### Crop-Related Predictions
- Crop recommendations based on conditions
- Optimal harvest timing
- Best selling periods

### Threat Assessment
- Possible pest identification
- Real-time threat detection
- Weather-related risks

### Irrigation Management
- Irrigation frequency recommendations
- Water management strategies
- Moisture-based guidance

## 🔍 Data Sources

### Sensor Data
- **Temperature**: Field temperature in °C
- **Humidity**: Ambient humidity percentage
- **Soil Moisture**: Soil moisture content percentage
- **NPK Values**: Nitrogen, Phosphorus, Potassium levels in ppm

### Weather Data
- **Current Conditions**: Temperature, humidity, wind speed, weather description
- **Forecast**: 3-day weather predictions
- **Location**: Mandi, Himachal Pradesh, India

## 🎯 Use Cases

### For Farmers
- **Crop Planning**: Get AI recommendations for optimal crop selection
- **Resource Management**: Optimize fertilizer and irrigation usage
- **Risk Mitigation**: Early detection of potential threats
- **Market Timing**: Optimal harvest and selling recommendations

### For Agricultural Consultants
- **Data Analysis**: Comprehensive soil and weather analysis
- **Recommendation Engine**: AI-powered agricultural advice
- **Monitoring**: Real-time field condition tracking

### For Researchers
- **Data Collection**: Structured agricultural data gathering
- **Pattern Analysis**: Weather and soil correlation studies
- **Model Validation**: LLM performance in agricultural contexts

## 🔧 Configuration

### Environment Variables
```bash
GROQ_API_TOKEN=your_token_here
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
```

### Model Configuration
- **Default Model**: DeepSeek R1 Distill Llama 70B
- **Temperature**: 0.7 (configurable)
- **Top-p**: 0.5 (configurable)
- **Max Retries**: 3 attempts with 3-second delays

### Caching
- **HTTP Cache**: 15-minute cache for weather data
- **Cache Location**: `http_cache.sqlite`

## 📈 Performance Features

### Optimization
- **Request Caching**: Reduces API calls and improves response times
- **Error Handling**: Robust error recovery with fallback mechanisms
- **Async Processing**: Non-blocking LLM API calls
- **Memory Management**: Efficient data handling and cleanup

### Monitoring
- **Event Logging**: Key events logged to `key_events.log`
- **Error Tracking**: Comprehensive error handling and reporting
- **Performance Metrics**: Response time monitoring

## 🛡️ Security & Privacy

- **API Token Security**: Environment variable-based configuration
- **Data Privacy**: No persistent storage of sensitive agricultural data
- **Input Validation**: JSON schema validation for all LLM responses
- **Error Sanitization**: Safe error message handling

## 🔄 API Integration

### Groq API
- **Endpoint**: `https://api.groq.com/openai/v1/chat/completions`
- **Authentication**: Bearer token
- **Rate Limiting**: Built-in retry mechanism
- **Response Format**: OpenAI-compatible JSON

### Weather API
- **Provider**: wttr.in
- **Location**: Mandi, Himachal Pradesh
- **Format**: JSON
- **Update Frequency**: 15-minute cache

## 🐛 Troubleshooting

### Common Issues

1. **API Token Error**
   ```
   Solution: Verify GROQ_API_TOKEN environment variable is set correctly
   ```

2. **Weather Data Unavailable**
   ```
   Solution: Check internet connection and wttr.in service status
   ```

3. **LLM Response Errors**
   ```
   Solution: Check API quota and model availability
   ```

4. **Port Conflicts**
   ```
   Solution: Gradio will automatically find an available port
   ```

### Debug Mode
Enable detailed logging by checking the `key_events.log` file for system events and errors.

## 📝 Development

### Project Structure
```
Smart-Agriculture-System-Using-LLMS/
├── dp_smart_agri.py          # Main application file
├── http_cache.sqlite         # HTTP cache database
├── key_events.log           # Event logging
└── README.md               # This file
```

### Code Organization
- **Configuration**: Environment variables and constants
- **Data Processing**: Sensor and weather data handling
- **AI Integration**: LLM API calls and response processing
- **UI Components**: Gradio interface and visualizations
- **Utilities**: Helper functions and formatters

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the event logs for debugging information

## 🔮 Future Enhancements

- **Multi-location Support**: Expand beyond Mandi, Himachal Pradesh
- **Historical Data**: Add data persistence and trend analysis
- **Mobile App**: Native mobile application
- **IoT Integration**: Direct sensor data integration
- **Advanced Analytics**: Machine learning model integration
- **Multi-language Support**: Localization for different regions

---

**Built with ❤️ for Smart Agriculture** 