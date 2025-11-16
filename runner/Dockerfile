FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm ci

COPY . .

# Set environment variables with defaults that can be overridden
ARG ALLOWED_ORIGINS=http://localhost:3000,https://music-ai-assistant-frontend.onrender.com,https://music-ai-assistant-backend.onrender.com
ENV ALLOWED_ORIGINS=${ALLOWED_ORIGINS}

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:5001/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

CMD ["npm", "start"]