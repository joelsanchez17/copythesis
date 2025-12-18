
#!/bin/bash

# Copy the coinhsl.tar.gz file to the home directory (assuming it's in the current directory)
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git ~/ThirdParty-HSL
cp coinhsl.tar.gz ~/coinhsl.tar.gz

# Change directory to ThirdParty-HSL and unpack coinhsl.tar.gz
cd ~/ThirdParty-HSL
gunzip ~/coinhsl.tar.gz
tar xf ~/coinhsl.tar

# Run the configure script
./configure

# Build using make
make

# Install the build
make install

# Export LD_LIBRARY_PATH (this will only apply for the current script's execution environment)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
