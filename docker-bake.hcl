// This file defines Docker build targets that can be built with:
// docker buildx bake [target]
//
// The Dockerfiles that define xngin-railway and xngin-worker-railway are very similar. Using a multi-stage build
// would reduce the boilerplate, but we use Railway's deployment system and it does not support selecting a single
// target in a Dockerfile. Also, Railway's Docker cache mounts require embedding the service's ID in the cache mount
// paths, and this cannot be parameterized. Railway's builders also do not support type=bind mounts.

// Default target when running `docker buildx bake`
group "default" {
  targets = [
    "xngin",
    "xngin-railway",
    "xngin-worker-railway",
  ]
}

// Base target with common settings
target "common" {
  context = "."
  platforms = ["linux/amd64"]
}

// Standard xngin image (equivalent to `docker build -t xngin .`)
target "xngin" {
  inherits = ["common"]
  tags = ["xngin:latest"]
}

// Railway-specific image (equivalent to `docker build -t xngin-railway -f Dockerfile.railway .`)
target "xngin-railway" {
  inherits = ["common"]
  dockerfile = "Dockerfile.railway"
  tags = ["xngin-railway:latest"]
}

// Railway-specific image (equivalent to `docker build -t xngin-worker-railway -f Dockerfile.worker.railway .`)
target "xngin-worker-railway" {
  inherits = ["common"]
  dockerfile = "Dockerfile.worker.railway"
  tags = ["xngin-worker-railway:latest"]
}
