#!/bin/bash

BS="\\\\"

cat biblio.bib \
  | sed -r -e "s|$BS([{}_$])|\\1|g" \
  | sed -r -e "s|{${BS}textbackslash}|$BS|g" \
  | sed -r -e "s|{${BS}textasciicircum}|^|g" \
  | sed -r -e "s|{${BS}textasciitilde}|$$${BS}sim$|g" \
  | sed -r -e "s|([ \\({]){([a-zA-Z]+)}|\\1\\2|g" \
  | sed -r -e "s|$BS$BS%|$BS%|g" \
  | sed -r -e "s|${BS}ensuremath\\{${BS}pi\\}|${BS}pi|g" \
  | sed -n -r -e "/abstract =.*/,/.*=.*/!p" \
  | grep -v "file = " \
  > biblio_fixed.bib

# Add pdg 2024
cat >>  biblio_fixed.bib <<EOF
@article{ParticleDataGroup:2024cfk,
    author = "Navas, S. and others",
    collaboration = "Particle Data Group",
    title = "{Review of particle physics}",
    doi = "10.1103/PhysRevD.110.030001",
    journal = "Phys. Rev. D",
    volume = "110",
    number = "3",
    pages = "030001",
    year = "2024"
}
EOF
