cd $2
for file in `find -type f -regex ".*\.\(frag\|vert\|rchit\|rmiss\|rgen\)"`
do
    echo "compiling $file"
    $1 "$file" -o "$file".spv
done
