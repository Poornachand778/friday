-- Friday AI - PostgreSQL Initialization
-- ======================================
-- Run automatically by Docker Compose on first start.
-- SQLAlchemy manages the full schema via db/schema.py;
-- this script just enables required extensions.

-- Enable pgvector for embedding search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for fuzzy text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable uuid-ossp for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
