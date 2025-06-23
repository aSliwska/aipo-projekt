set -e

echo "Cleaning up existing Tesseract and OpenALPR installations..."

sudo rm -f /usr/local/bin/tesseract
sudo rm -f /usr/local/lib/libtesseract*
sudo rm -rf /usr/local/include/tesseract
sudo rm -rf /usr/local/share/tessdata
sudo rm -f /usr/local/lib/pkgconfig/tesseract.pc
sudo rm -f /usr/local/bin/alpr
sudo rm -f /usr/local/bin/alprd
sudo rm -f /usr/local/bin/openalpr-utils-*
sudo rm -f /usr/local/lib/libopenalpr*
sudo rm -rf /usr/local/include/alpr*
sudo rm -rf /usr/local/share/openalpr
sudo rm -rf /etc/openalpr
sudo rm -f /usr/local/lib/pkgconfig/openalpr.pc
sudo rm -rf /tmp/tesseract-3.04.01
sudo rm -rf /tmp/openalpr-build

sudo ldconfig

sudo apt-get update

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    liblog4cplus-dev \
    libcurl4-openssl-dev

sudo apt-get install -y \
    libopencv-dev \
    libopencv-contrib-dev \
    opencv-data

sudo apt-get install -y \
    libleptonica-dev \
    libtesseract-dev \
    tesseract-ocr-all \
    autotools-dev \
    autoconf \
    automake \
    libtool \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev

cd /tmp

wget -q https://github.com/tesseract-ocr/tesseract/archive/3.04.01.tar.gz
tar -xzf 3.04.01.tar.gz
cd tesseract-3.04.01

./autogen.sh
./configure --prefix=/usr/local

make -j$(nproc)

sudo make install

sudo ldconfig

hash -r
if /usr/local/bin/tesseract --version 2>&1 | grep -q "3.04.01"; then
    echo "✓ Tesseract 3.04.01 installed successfully"
else
    echo "✗ Tesseract installation failed"
    exit 1
fi

OPENALPR_DIR="/tmp/openalpr"
if [ ! -d "$OPENALPR_DIR" ]; then
    echo "Cloning OpenALPR repository..."
    git clone https://github.com/openalpr/openalpr.git "$OPENALPR_DIR"
fi

cd "$OPENALPR_DIR"

# Create build directory
mkdir -p build
cd build

cmake ../src \
    -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
    -DCMAKE_BUILD_TYPE=RelWithDebugInfo \
    -DWITH_TESTS=ON \
    -DWITH_UTILITIES=ON

make -j$(nproc)

sudo make install

sudo ldconfig

hash -r
if alpr --version 2>&1 | grep -q "2.3.0"; then
    echo "✓ OpenALPR 2.3.0 installed successfully"
else
    echo "✗ OpenALPR installation failed"
    exit 1
fi

if ldd /usr/local/bin/alpr | grep -q "libtesseract.so.3"; then
    echo "✓ OpenALPR correctly linked to Tesseract 3.x"
else
    echo "✗ OpenALPR not properly linked to Tesseract 3.x"
    exit 1
fi


cd /
rm -rf /tmp/tesseract-3.04.01*
rm -rf /tmp/openalpr

echo "Installation script completed successfully!"