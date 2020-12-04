dirs=$(ls -d */)
rm -f remove_nonconverge_vasp.log
for dir in $dirs; do
    if [ -f $dir"OUTCAR" ]; then
        if test -z "$(grep "EDIFF is reached" $dir"OUTCAR")"; then
            echo removed $dir
            echo removed $dir >>remove_nonconverge_vasp.log 
            rm -rf $dir           
        fi
    fi
done
