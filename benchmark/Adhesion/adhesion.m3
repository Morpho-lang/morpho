// Adhesion test 3.
// Now add in an area constraint
// Progressively fix vertices with a level set constraint

etol = 1e-8

mfloor = cuboid({-1,-1,-1},{1,1,-1})

m=manifoldload("Adhesion/sphere.mesh")
m[1].orient()
//m.refine()
//m.refine()
//m[1].equiangulate()
//m[1].export("disk-fixed.mesh")

mref=manifoldload("Adhesion/sphere.mesh")
//mref[1].orient()
//mref.refine()
//mref.refine()
//mref[1].equiangulate()

normalize(list)={
  norm=0;
  for (i=1, i<=length(list), i=i+1,
    norm=norm+list[i]**2
  );
  return list/sqrt(norm)
}

// Linear elasticity for 3D
ll=new(linearelastic3d)
ll.setreference(mref[1])
m.addenergy(ll)

//This needs to be changed for volume
//la=new(enclosedarea)
//bnd=m[1].selectboundary()
//la=new(surfacetension)
//la.settarget(3.13836)
//m.addconstraint(la)

//la.total(m[1])

lv = new(volume)
v0 = lv.total(m[1])
lv.settarget(v0)
//m.addconstraint(lv)

print v0

// Level set constraint on the floor
y0=1
llower=new(levelset);
llower.setcoordinates({x,y,z});
llower.setexpression(z+y0);
llower.setgradient({0,0,1});

// Now move
attach=m.select(z<-0.9, coordinates={x,y,z}, grade=0)
attach.idlistforgrade(m[1], 0)

m.addlocalconstraint(llower, attach)

m.totalenergy()

vis(m) = {
  g = m.draw();
  g.setviewdirection({1,1,0});
  g.setvertical({0,0,1});
  g.join(mfloor.draw())
}

//s=show(vis(m))

relax(n, sc) = {
  eold=m.totalenergy()
  de=1
  for(i=0, i<n && abs(de)>etol, i=i+1, {
    m.relax(scale=sc/2);
    enew=m.totalenergy()
    de=eold-enew
    print("Iteration ", i, " total: ", enew, " de: ", de, " vol: ", lv.total(m[1]));
    eold=enew
  }
  )
}

//ssel=show(plotselection(attach).join(mfloor.draw()));

redoconstraint(xthresh, ythresh) = {
  m.removefunctional(llower); //Remove existing constraint
  //Select verticies less than y=-0.99 or (verticies less than y=-0.98 and x values greater than the existing contact radius)
  newattach=m.select(y<-0.99 || (y<ythresh && abs(x)>xthresh) || (y<ythresh && abs(z)>xthresh), coordinates={x,y,z}, grade=0);
  ssel.update(plotselection(newattach));
  m.addlocalconstraint(llower, newattach);
  newattach
}

finda(attach) = {
  a = 0;
  ids = attach.idlistforgrade(m, 0);
  for (i=1, i<=length(ids), i=i+1, {
    x = m[1].vertexposition(ids[i]);
    if(abs(x[1])>a, a = abs(x[1])); //x coord
    if(abs(x[3])>a, a = abs(x[3])); //z coord
  });
  a
}


findnexty() = {
  y0=0;
  sb=m[1].selectboundary(); //Select the edge of the mesh
  ids = sb.idlistforgrade(m, 0); //list of indicies for verticies on the edge of the disk
  //This loop gets the lowest y value above -0.99
  for (i=1, i<=length(ids), i=i+1, {
    x = m[1].vertexposition(ids[i]); //Get actual vertex, {x,y,z}, from an index
    if(x[2]<y0 && x[2]>-0.99, y0=x[2]);
  });
  y0
}

//returns a list of y values for verticies associated with the constraint attach
getConyvals(attach) = {
  ids = attach.idlistforgrade(m, 0);
  table({x = m[1].vertexposition(ids[i]);
         x[2]
        },{i,length(ids)})
}

stepsize=0.1
relaxSteps = 50

relax(relaxSteps, stepsize)

quit
