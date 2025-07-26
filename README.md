# 🤖 Free Job Application Agent - No OpenAI Costs!

**Intelligent Automation System for LinkedIn Easy Apply Jobs - Powered by Free Local AI**

This is a modified version of the original ApplyAgent.AI that uses **LocalAI** instead of OpenAI, eliminating API costs while maintaining full functionality.

## 🆓 What Makes This Free?

- **LocalAI Integration**: Uses open-source LocalAI as a drop-in replacement for OpenAI
- **No API Costs**: Run everything locally without paying for external AI services
- **Same Functionality**: All original features work exactly the same
- **Easy Setup**: Simple Docker Compose deployment

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd job_application_agent_free
   ```

2. **Start LocalAI and the Job Agent**
   ```bash
   docker-compose -f docker-compose.localai.yml up -d
   ```

3. **Wait for LocalAI to download models** (first run takes a few minutes)

4. **Access the application**
   - Job Agent: http://localhost:8501
   - LocalAI API: http://localhost:8080

### Option 2: Manual Setup

1. **Install LocalAI**
   ```bash
   # Using Docker
   docker run -p 8080:8080 --name localai -ti quay.io/go-skynet/local-ai:latest
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env to set USE_LOCALAI=true
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## ⚙️ Configuration

### Environment Variables

Key settings in your `.env` file:

```bash
# Enable LocalAI
USE_LOCALAI=true
LOCALAI_URL=http://localhost:8080
LOCALAI_MODEL=gpt-3.5-turbo

# Fallback if LocalAI is unavailable
FALLBACK_PROVIDER=openai
```

### Model Selection

LocalAI supports various open-source models:
- `gpt-3.5-turbo` (default, good balance of speed/quality)
- `gpt-4` (higher quality, slower)
- Custom models (configure in `preload_models.yaml`)

## 🔧 How It Works

1. **LocalAI Server**: Runs locally and provides OpenAI-compatible API
2. **Modified Agent**: Uses LocalAI instead of OpenAI for all AI operations
3. **Same Interface**: Streamlit UI remains unchanged
4. **Browser Automation**: Playwright handles LinkedIn interactions

## 📊 Performance Comparison

| Feature | Original (OpenAI) | Free Version (LocalAI) |
|---------|------------------|------------------------|
| Cost | $0.01-0.10 per application | $0.00 |
| Speed | Fast (cloud) | Moderate (local) |
| Privacy | Data sent to OpenAI | Data stays local |
| Setup | API key required | Docker setup |
| Quality | High | Good (model dependent) |

## 🛠️ Troubleshooting

### LocalAI Not Starting
- Check Docker is running
- Ensure port 8080 is available
- Check logs: `docker logs localai`

### Models Not Loading
- First run downloads models (can take 10-30 minutes)
- Check internet connection
- Verify `preload_models.yaml` configuration

### Agent Not Connecting to LocalAI
- Verify `LOCALAI_URL` in environment
- Check LocalAI health: `curl http://localhost:8080/health`
- Review application logs

## 🔄 Switching Back to OpenAI

To use OpenAI instead of LocalAI:

1. Set `USE_LOCALAI=false` in `.env`
2. Add your OpenAI API key: `OPENAI_API_KEY=your_key`
3. Restart the application

## 📝 Model Customization

To use different models:

1. Edit `preload_models.yaml`
2. Add model URLs from Hugging Face
3. Restart LocalAI container
4. Update `LOCALAI_MODEL` in `.env`

## 🤝 Contributing

This free version maintains compatibility with the original project. Contributions welcome for:
- Additional model support
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📄 License

Same as original project - MIT License

## 🙏 Acknowledgments

- Original ApplyAgent.AI project
- LocalAI community
- Open-source LLM developers

---

**Enjoy free job applications! 🎉**

