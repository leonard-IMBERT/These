#!/bin/bash

BS="\\\\"

cat biblio.bib \
  | sed -r -e "s|$BS([{}_$])|\\1|g" \
  | sed -r -e "s|{${BS}textbackslash}|$BS|g" \
  | sed -r -e "s|{${BS}textasciicircum}|^|g" \
  | sed -r -e "s|{${BS}textasciitilde}|$\\sim$|g" \
  | sed -r -e "s|([ \\({]){([a-zA-Z]+)}|\\1\\2|g" \
  | sed -r -e "s|$BS$BS%|$BS%|g" \
  > biblio_fixed.bib
