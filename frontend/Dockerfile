FROM node:18-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Set build-time environment variable
# This will be baked into the Next.js build
ARG NEXT_PUBLIC_API_URL=https://music-ai-assistant-backend.onrender.com
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Build the application (removing --turbopack as it's dev-only)
RUN npx next build

EXPOSE 3000

CMD ["npm", "start"]