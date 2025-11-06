FROM node:18-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

RUN npm run build

CMD ["npm", "start"]