COMPOSE ?= docker-compose
ENV_FILE ?= .env

.PHONY: build clean dev down

build:
	${COMPOSE} build frontend backend

clean: 
	${COMPOSE} down -v

dev:
	${COMPOSE} up -d frontend backend

down:
	${COMPOSE} down