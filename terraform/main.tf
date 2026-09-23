terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "ap-south-1"
}

variable "instance_type" {
  default = "t3.medium" # embeddings + reranker are CPU-bound; bump to t3.large under load
}

variable "key_name" {
  description = "Existing EC2 key pair name for SSH access"
  type        = string
}

variable "anthropic_api_key" {
  description = "Passed in as TF_VAR_anthropic_api_key, never committed"
  type        = string
  sensitive   = true
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "rag_platform" {
  name        = "rag-platform-sg"
  description = "Allow API traffic and SSH"

  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # tighten to an ALB security group / your IP in real deployments
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # tighten to your IP
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "rag_platform" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.rag_platform.id]

  user_data = <<-EOF
    #!/bin/bash
    set -e
    apt-get update -y
    apt-get install -y docker.io docker-compose-plugin git
    systemctl enable --now docker
    git clone https://github.com/YOUR_USERNAME/rag-platform.git /opt/rag-platform
    cd /opt/rag-platform
    echo "ANTHROPIC_API_KEY=${var.anthropic_api_key}" > .env
    echo "VECTOR_STORE_BACKEND=qdrant" >> .env
    echo "CACHE_BACKEND=redis" >> .env
    docker compose -f docker/docker-compose.yml up -d --build
  EOF

  tags = {
    Name = "rag-platform"
  }
}

output "public_ip" {
  value = aws_instance.rag_platform.public_ip
}

output "api_url" {
  value = "http://${aws_instance.rag_platform.public_ip}:8000"
}
