// This file defines Docker build targets that can be built with:
// docker buildx bake [target]

// Default target when running `docker buildx bake`
group "default" {
  targets = [
    "xngin",
    "xngin-railway",
    "xngin-snapshots-railway",
    "xngin-tq-railway",
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

// Railway-specific image (equivalent to `docker build -t xngin-snapshots-railway -f Dockerfile.snapshots.railway .`)
target "xngin-snapshots-railway" {
  inherits = ["common"]
  dockerfile = "Dockerfile.snapshots.railway"
  tags = ["xngin-snapshots-railway:latest"]
}

// Railway-specific image (equivalent to `docker build -t xngin-tq-railway -f Dockerfile.tq.railway .`)
target "xngin-tq-railway" {
  inherits = ["common"]
  dockerfile = "Dockerfile.tq.railway"
  tags = ["xngin-tq-railway:latest"]
}
