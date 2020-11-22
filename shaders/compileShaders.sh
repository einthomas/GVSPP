cd $1
for file in `find -type f -regex ".*\.\(frag\|vert\|rchit\|rmiss\|rgen\|comp\)"`
do
    echo "compiling $file"
    $VULKAN_SDK/bin/glslc --target-spv=spv1.5 "$file" -o "$file".spv
done
