FROM node:18-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Build the application (removing --turbopack as it's dev-only)
RUN npx next build

EXPOSE 3000

CMD ["npm", "start"]