# Reference: https://github.com/openai/mujoco-py/issues/682
mkdir -p $HOME/.mujoco/mujoco210
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include

mkdir -p $HOME/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.*.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
sudo ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.*.dylib /usr/local/lib/

# For M1 (arm64) mac users:
# brew install glfw # if you don't have it installed
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

# which python
# exit

export CC=/opt/homebrew/bin/gcc-12         # see https://github.com/openai/mujoco-py/issues/605
pip install mujoco-py && python -c 'import mujoco_py'
