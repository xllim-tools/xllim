# ==============================================================================
# GLOBAL ARGUMENTS
# ==============================================================================
# "SOURCE" controls whether to build from scratch ('local') or use CI artifacts ('ci')
ARG SOURCE=local

# ==============================================================================
# STAGE 1: Builder (Manylinux 2.28)
# Used only if SOURCE=local. Provides an isolated, standard build environment.
# ==============================================================================
FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

WORKDIR /app

RUN dnf install -y blas-devel lapack-devel armadillo-devel

ENV CXXFLAGS="-fPIC" \
    CFLAGS="-fPIC" \
    CMAKE_ARGS="-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_SHARED_LINKER_FLAGS='-static-libstdc++' \
        -DCMAKE_MODULE_LINKER_FLAGS='-static-libstdc++' \
        -DBoost_ROOT=/usr/local \
        -DBoost_NO_SYSTEM_PATHS=ON \
        -DBoost_USE_STATIC_LIBS=ON \
        -DARMADILLO_INCLUDE_DIR=/usr/include"

# Build and install Boost from source (specific version requirement)
RUN curl -L https://archives.boost.io/release/1.78.0/source/boost_1_78_0.tar.gz -o boost.tar.gz \
    && tar -xzf boost.tar.gz \
    && cd boost_1_78_0 \
    && ./bootstrap.sh --prefix=/usr/local \
    && ./b2 --with-system --with-thread --with-random link=static variant=release cxxflags="-fPIC" cflags="-fPIC" -j$(nproc) install \
    && cd .. \
    && rm -rf boost_1_78_0 boost.tar.gz

# Set Python 3.11 as the default build environment
ENV PATH="/opt/python/cp311-cp311/bin:${PATH}"

# Copy project source code
COPY . .

# 1. Build the Source Distribution (sdist)
# 2. Generate the binary Wheel
# 3. Use 'auditwheel' to bundle external shared libraries into the wheel
RUN python -m build --sdist --outdir dist \
    && pip wheel dist/*.tar.gz --no-deps -w /tmp/wheels \
    && for whl in /tmp/wheels/*.whl; do \
        auditwheel repair "$whl" --plat manylinux_2_28_x86_64 -w /internal_wheelhouse; \
    done

# ==============================================================================
# STAGE 2 & 3: Source Redirection (The Pivot)
# ==============================================================================

# Option A: Local build (grab wheels from the builder stage)
FROM scratch AS wheels-source-local
COPY --from=builder /internal_wheelhouse/*.whl /wheels/

# Option B: CI build (grab wheels from the local host 'wheelhouse' directory)
FROM scratch AS wheels-source-ci
COPY wheelhouse/*.whl /wheels/

# Final selection stage
FROM wheels-source-${SOURCE} AS wheels_to_install

# ==============================================================================
# STAGE 4: Final Production Image
# Lightweight runtime environment based on Debian Slim
# ==============================================================================
FROM python:3.11-slim AS final

WORKDIR /app

# Copy wheels from the selected source
COPY --from=wheels_to_install /wheels /wheels

# Install the package and cleanup
RUN pip install --upgrade pip \
    && pip install --no-cache-dir /wheels/*cp311*.whl \
    && rm -rf /wheels

# Verify installation on startup
CMD ["python", "-c", "import xllim; print(xllim.__version__)"]