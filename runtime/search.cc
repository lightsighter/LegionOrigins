
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

namespace Config {
  int num_tree_nodes = 100;
  int num_objects = 1000;
  int num_rays = 100;
  int partition_size = 100; // 0 = no partitioning
  bool visible_partitions = true;
  int random_seed = 12345;
  bool args_read = false;
};

#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN 
enum {
  TASKID_CREATE_TREE = (TASK_ID_AVAILABLE+0),
  TASKID_ADD_OBJECTS,
  TASKID_FIND_ISECT,
};

// keep things simple by having spheres as the only object - they do 
//  have fun reflectance properties...
struct Object {
  double p[3], r;
  ptr_t<Object> next;
};

struct Ray {
  double p[3];  // ray's starting point
  double dp[3]; // ray's direction vector
};

// a KD-tree splits a volume along one axis-aligned plane at each
//  node - keep a left and right pointer, as well as a possible new
//  (sub)region for each side
// also keep a list of local objects
struct KDnode {
  int axis;
  double cutoff;
  ptr_t<KDnode> left, right;
  LogicalHandle r_left, r_right;
  ptr_t<Object> objects;
};

typedef std::pair<ptr_t<KDnode>, LogicalHandle> CreateTreeResult;

struct IntersectQuery {
  LogicalHandle r_tree, r_objects;
  ptr_t<KDnode> start;
  Ray ray;
};

typedef std::pair<double, ptr_t<Object> > IntersectResult;

template<AccessorType AT>
IntersectResult find_isect_task(const void *args, size_t arglen, 
				const std::vector<PhysicalRegion<AT> > &regions,
				Context ctx, HighLevelRuntime *runtime)
{
  PhysicalRegion<AT> r_tree = regions[0];
  PhysicalRegion<AT> r_objects = regions[1];

  printf("HEREHERHEHRE\n");

  IntersectQuery q = *(IntersectQuery *)args;
  Ray& ray = q.ray;

  KDnode n = r_tree.read(q.start);

  IntersectResult r;
  r.first = 1e6;  // something large
  r.second = ptr_t<Object>::nil();

  // have to check all objects in the current kd node no matter what
  ptr_t<Object> optr = n.objects;
  while(optr) {
    Object o = r_objects.read(optr);

    // vector math to find closest point of approach to sphere
    double ps[3];
    ps[0] = o.p[0] - ray.p[0];
    ps[1] = o.p[1] - ray.p[1];
    ps[2] = o.p[2] - ray.p[2];

    double ps2 = (ps[0]*ps[0] + ps[1]*ps[1] + ps[2]*ps[2]);
    double pr = (ps[0]*ray.dp[0] + ps[1]*ray.dp[1] + ps[2]*ray.dp[2]);
    double rs2 = ps2 - pr*pr;

    // rs2 is tangeant radius (squared) if greater than r^2, no intersection
    if(rs2 <= (o.r*o.r)) {
      // back up pr to actual point of intersection with circle of radius r
      pr -= sqrt(o.r*o.r - rs2);

      // is this better than any intersection we've seen so far?
      if((pr > 0) && (pr < r.first)) {
	r.first = pr;
	r.second = optr;
      }
    }

    optr = o.next;
  }

  // now check against any child nodes
  if(n.axis < 0) return r;

  // positive direction vector component means it is going 'left' to 'right'
  if(ray.dp[n.axis] > 0) {
    // check left if it exists and ray starts before cutoff
    if(n.left && (ray.p[n.axis] < n.cutoff)) {
      q.start = n.left;
      IntersectResult r2;
      // if we have a region annotation, we have to respawn
      if(n.r_left.exists()) {
	std::vector<RegionRequirement> isect_regions;
	isect_regions.push_back(RegionRequirement(q.r_tree,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_tree));
	isect_regions.push_back(RegionRequirement(q.r_objects,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_objects));
	isect_regions.push_back(RegionRequirement(n.r_left,
						  NO_ACCESS, NO_MEMORY, EXCLUSIVE,
						  n.r_left));
	Future f = runtime->execute_task(ctx, TASKID_FIND_ISECT,
					 isect_regions, &q, sizeof(q), true);
	r2 = f.get_result<IntersectResult>();
      } else {
	// if not, we can just recurse
	r2 = find_isect_task(&q, sizeof(q), regions, ctx, runtime);
      }

      if(r2.second) {
	// we got something - the winner will either be it or our local
	//  match (if any)
	if(r.second && (r.first < r2.first))
	  return r;
	return r2;
      }
    }

    // try other side
    if(n.right) {
      q.start = n.right;
      IntersectResult r2;
      // if we have a region annotation, we have to respawn
      if(n.r_right.exists()) {
	std::vector<RegionRequirement> isect_regions;
	isect_regions.push_back(RegionRequirement(q.r_tree,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_tree));
	isect_regions.push_back(RegionRequirement(q.r_objects,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_objects));
	isect_regions.push_back(RegionRequirement(n.r_right,
						  NO_ACCESS, NO_MEMORY, EXCLUSIVE,
						  n.r_right));
	Future f = runtime->execute_task(ctx, TASKID_FIND_ISECT,
					 isect_regions, &q, sizeof(q), true);
	r2 = f.get_result<IntersectResult>();
      } else {
	// if not, we can just recurse
	r2 = find_isect_task(&q, sizeof(q), regions, ctx, runtime);
      }

      if(r2.second) {
	// we got something - the winner will either be it or our local
	//  match (if any)
	if(r.second && (r.first < r2.first))
	  return r;
	return r2;
      }
    }

    // if we get here, our local match (if any) is the only choice
    return r;
  } else {
    // same as above, but swapping left and right

    // check right if it exists and ray starts before cutoff
    if(n.right && (ray.p[n.axis] > n.cutoff)) {
      q.start = n.right;
      IntersectResult r2;
      // if we have a region annotation, we have to respawn
      if(n.r_right.exists()) {
	std::vector<RegionRequirement> isect_regions;
	isect_regions.push_back(RegionRequirement(q.r_tree,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_tree));
	isect_regions.push_back(RegionRequirement(q.r_objects,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_objects));
	isect_regions.push_back(RegionRequirement(n.r_right,
						  NO_ACCESS, NO_MEMORY, EXCLUSIVE,
						  n.r_right));
	Future f = runtime->execute_task(ctx, TASKID_FIND_ISECT,
					 isect_regions, &q, sizeof(q), true);
	r2 = f.get_result<IntersectResult>();
      } else {
	// if not, we can just recurse
	r2 = find_isect_task(&q, sizeof(q), regions, ctx, runtime);
      }

      if(r2.second) {
	// we got something - the winner will either be it or our local
	//  match (if any)
	if(r.second && (r.first < r2.first))
	  return r;
	return r2;
      }
    }

    // try other side
    if(n.left) {
      q.start = n.left;
      IntersectResult r2;
      // if we have a region annotation, we have to respawn
      if(n.r_left.exists()) {
	std::vector<RegionRequirement> isect_regions;
	isect_regions.push_back(RegionRequirement(q.r_tree,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_tree));
	isect_regions.push_back(RegionRequirement(q.r_objects,
						  READ_ONLY, NO_MEMORY, EXCLUSIVE,
						  q.r_objects));
	isect_regions.push_back(RegionRequirement(n.r_left,
						  NO_ACCESS, NO_MEMORY, EXCLUSIVE,
						  n.r_left));
	Future f = runtime->execute_task(ctx, TASKID_FIND_ISECT,
					 isect_regions, &q, sizeof(q), true);
	r2 = f.get_result<IntersectResult>();
      } else {
	// if not, we can just recurse
	r2 = find_isect_task(&q, sizeof(q), regions, ctx, runtime);
      }

      if(r2.second) {
	// we got something - the winner will either be it or our local
	//  match (if any)
	if(r.second && (r.first < r2.first))
	  return r;
	return r2;
      }
    }

    // if we get here, our local match (if any) is the only choice
    return r;
  }
}

extern RegionRuntime::LowLevel::Logger::Category log_app;

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen, const std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  printf("Running top level task\n");

  while(!Config::args_read)
    sleep(1);

  // first step, build regions for the tree nodes and the objects we'll create
  LogicalHandle r_tree = runtime->create_logical_region<KDnode>(ctx,
								 Config::num_tree_nodes);

  // spawn a task to build the tree
  CreateTreeResult ctr;
  ptr_t<KDnode> root;
  {
    std::vector<RegionRequirement> build_regions;
    build_regions.push_back(RegionRequirement(r_tree, 
					      READ_WRITE, ALLOCABLE, EXCLUSIVE,
					      r_tree));
    Future f = runtime->execute_task(ctx, TASKID_CREATE_TREE, 
				     build_regions,
				     &r_tree, sizeof(r_tree), true);
    ctr = f.get_result<CreateTreeResult>();
    root = ctr.first;
  }

  printf("root = %d, region=%x\n", root.value, ctr.second.id);

  LogicalHandle r_objects = runtime->create_logical_region<Object>(ctx,
								   Config::num_objects);
  
  {
    std::vector<RegionRequirement> addobj_regions;
    addobj_regions.push_back(RegionRequirement(r_tree,
					       READ_WRITE, NO_MEMORY, EXCLUSIVE,
					       r_tree));
    addobj_regions.push_back(RegionRequirement(r_objects,
					       READ_WRITE, ALLOCABLE, EXCLUSIVE,
					       r_objects));
    Future f = runtime->execute_task(ctx, TASKID_ADD_OBJECTS,
				     addobj_regions,
				     &root, sizeof(root), true);
    f.get_void_result();
  }

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  DetailedTimer::clear_timers();

  std::vector<Future> futures;
  for(int i = 0; i < Config::num_rays; i++) {
    // create an arbitrary ray and do an intersection test
    Ray ray;
    ray.p[0] = 2*drand48() - 1;
    ray.p[1] = 2*drand48() - 1;
    ray.p[2] = 2*drand48() - 1;
    ray.dp[0] = 2*drand48() - 1;
    ray.dp[1] = 2*drand48() - 1;
    ray.dp[2] = 2*drand48() - 1;

    // clamp to one side of the cube and make dir vector go inward
    int a = int(drand48() * 3);
    if(drand48() < 0.5) {
      ray.p[a] = -1;
      ray.dp[a] = 1;
    } else {
      ray.p[a] = 1;
      ray.dp[a] = -1;
    }
    double dp_len = sqrt(ray.dp[0]*ray.dp[0] + ray.dp[1]*ray.dp[1] + ray.dp[2]*ray.dp[2]);
    ray.dp[0] /= dp_len;
    ray.dp[1] /= dp_len;
    ray.dp[2] /= dp_len;

    IntersectQuery q;
    q.start = root;
    q.ray = ray;
    q.r_tree = r_tree;
    q.r_objects = r_objects;

    std::vector<RegionRequirement> isect_regions;
#if 0
    isect_regions.push_back(RegionRequirement(r_tree,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      r_tree));
    isect_regions.push_back(RegionRequirement(r_objects,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      r_objects));
#else
    isect_regions.push_back(RegionRequirement(r_tree,
					      READ_ONLY, NO_MEMORY, RELAXED,
					      r_tree));
    isect_regions.push_back(RegionRequirement(r_objects,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      r_objects));
    isect_regions.push_back(RegionRequirement(ctr.second,
					      NO_ACCESS, NO_MEMORY, RELAXED,
					      ctr.second));
#endif
    Future f = runtime->execute_task(ctx, TASKID_FIND_ISECT,
				     isect_regions, &q, sizeof(q), true);
    futures.push_back(f);
  }

  // now wait for all queries to finish
  int hit_count = 0;
  while(futures.size() > 0) {
    Future f = futures.back();
    futures.pop_back();
    IntersectResult r = f.get_result<IntersectResult>();
    if(r.second)
      hit_count++;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
		     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  DetailedTimer::report_timers();

  log_app.info("all done - got %d hits\n", hit_count);
}

template<AccessorType AT>
void add_nodes_to_set(PhysicalRegion<AT> r_tree,
		      ptr_t<KDnode> nptr,
		      std::set<ptr_t<KDnode> >& colorset)
{
  colorset.insert(nptr);

  KDnode n = r_tree.read(nptr);

  if(n.left && !n.r_left.exists())
    add_nodes_to_set(r_tree, n.left, colorset);

  if(n.right && !n.r_right.exists())
    add_nodes_to_set(r_tree, n.right, colorset);
}

template<AccessorType AT>
int group_tree_nodes(PhysicalRegion<AT> r_tree,
		     LogicalHandle rl_tree,
		     ptr_t<KDnode> nptr,
		     int max_count,
		     std::vector<std::set<ptr_t<KDnode> > >& coloring,
		     std::vector<ptr_t<KDnode> >& colorstarts,
		     std::vector<bool>& colorstart_sides)
{
  KDnode n = r_tree.read(nptr);

  // roll up counts on the left and right - if either exceeds the limit, 
  //  give them a color
  int left_count;
  if(n.left) {
    left_count = group_tree_nodes(r_tree, rl_tree, n.left, max_count,
				  coloring, colorstarts, colorstart_sides);
    if(left_count >= max_count) {
      colorstarts.push_back(nptr);
      colorstart_sides.push_back(true);
      coloring.push_back(std::set<ptr_t<KDnode> >());
      add_nodes_to_set(r_tree, n.left, coloring.back());
      left_count = 0;
      n.r_left = rl_tree; // placeholder - stops the add_nodes_to_set descent
    }
  } else
    left_count = 0;

  int right_count;
  if(n.right) {
    right_count = group_tree_nodes(r_tree, rl_tree, n.right, max_count,
				  coloring, colorstarts, colorstart_sides);
    if(right_count >= max_count) {
      colorstarts.push_back(nptr);
      colorstart_sides.push_back(false);
      coloring.push_back(std::set<ptr_t<KDnode> >());
      add_nodes_to_set(r_tree, n.right, coloring.back());
      right_count = 0;
      n.r_right = rl_tree; // placeholder - stops the add_nodes_to_set descent
    }
  } else
    right_count = 0;

  return (left_count + right_count + 1);
}

template<AccessorType AT>
ptr_t<KDnode> create_random_tree(PhysicalRegion<AT> r_tree,
				 LogicalHandle rl_tree,
				 double x_min, double x_max,
				 double y_min, double y_max,
				 double z_min, double z_max,
				 int node_count)
{
  ptr_t<KDnode> ptr = r_tree.template alloc<KDnode>();

  // we just used up one node
  node_count--;

  KDnode n;

  n.axis = -1;
  n.cutoff = 0;
  n.left = ptr_t<KDnode>::nil();
  n.right = ptr_t<KDnode>::nil();
  n.r_left = LogicalHandle::NO_REGION;
  n.r_right = LogicalHandle::NO_REGION;
  n.objects = ptr_t<Object>::nil();

  if(node_count > 0) {
    int l_count = int((node_count + 1) * drand48());
    int r_count = node_count - l_count;

    int a_split = int(3 * drand48());
    double fraction = drand48();

    n.axis = a_split;
    switch(a_split) {
    case 0:
      n.cutoff = x_min + fraction * (x_max - x_min);
      if(l_count)
	n.left = create_random_tree(r_tree, rl_tree,
				    x_min, n.cutoff,
				    y_min, y_max,
				    z_min, z_max,
				    l_count);
      if(r_count)
	n.right = create_random_tree(r_tree, rl_tree,
				     n.cutoff, x_max,
				     y_min, y_max,
				     z_min, z_max,
				     r_count);
      break;

    case 1:
      n.cutoff = y_min + fraction * (y_max - y_min);
      if(l_count)
	n.left = create_random_tree(r_tree, rl_tree,
				    x_min, x_max,
				    y_min, n.cutoff,
				    z_min, z_max,
				    l_count);
      if(r_count)
	n.right = create_random_tree(r_tree, rl_tree,
				     x_min, x_max,
				     n.cutoff, y_max,
				     z_min, z_max,
				     r_count);
      break;

    case 2:
      n.cutoff = z_min + fraction * (z_max - z_min);
      if(l_count)
	n.left = create_random_tree(r_tree, rl_tree,
				    x_min, x_max,
				    y_min, y_max,
				    z_min, n.cutoff,
				    l_count);
      if(r_count)
	n.right = create_random_tree(r_tree, rl_tree,
				     x_min, x_max,
				     y_min, y_max,
				     n.cutoff, z_max,
				     r_count);
      break;
    }
  }

  // HACK: annotate ALL pointers and see if we can survive the task spawn storm
  if(n.left) n.r_left = rl_tree;
  if(n.right) n.r_right = rl_tree;

  //printf("%d -> (%d/%f) %d, %d\n", ptr.value, n.axis, n.cutoff, n.left.value, n.right.value);
  r_tree.write(ptr, n);

  return ptr;
}

template<AccessorType AT>
CreateTreeResult create_tree_task(const void *args, size_t arglen, const std::vector<PhysicalRegion<AT> > &regions,
			       Context ctx, HighLevelRuntime *runtime)
{
  printf("Running create_tree_task\n");

  PhysicalRegion<AT> r_tree = regions[0];

  LogicalHandle rl_tree = *(LogicalHandle *)args;


  srand48(Config::random_seed);
  ptr_t<KDnode> tree = create_random_tree(r_tree, rl_tree,
					  -1, 1, -1, 1, -1, 1, 
					  Config::num_tree_nodes);

  std::vector<std::set<ptr_t<KDnode> > > coloring;
  std::vector<ptr_t<KDnode> > colorstarts;
  std::vector<bool> colorstart_sides;

  group_tree_nodes(r_tree, rl_tree, tree, Config::partition_size,
		   coloring, colorstarts, colorstart_sides);

  unsigned last_subr = coloring.size();
  coloring.push_back(std::set<ptr_t<KDnode> >());
  add_nodes_to_set(r_tree, tree, coloring.back());

  printf("creating %zd partitions\n", coloring.size());

  Partition<KDnode> part = runtime->create_partition(ctx, rl_tree,
						     coloring,
						     true);

  // if desired, go back and fix up the region pointers so we actually know
  //  which subregion we're crossing into
#if 1
  if(Config::visible_partitions) {
    for(unsigned i = 0; i < last_subr; i++) {
      LogicalHandle subr = runtime->get_subregion<KDnode>(ctx, part, i);
      KDnode n = r_tree.read(colorstarts[i]);
      if(colorstart_sides[i])
	n.r_left = subr;
      else
	n.r_right = subr;
      r_tree.write(colorstarts[i], n);
    }
  }
#endif

  return std::make_pair(tree, runtime->get_subregion<KDnode>(ctx, part, last_subr));
}

template<AccessorType AT>
void add_objects_task(const void *args, size_t arglen, const std::vector<PhysicalRegion<AT> > &regions,
			       Context ctx, HighLevelRuntime *runtime)
{
  printf("Running add_objects_task\n");

  PhysicalRegion<AT> r_tree = regions[0];
  PhysicalRegion<AT> r_objects = regions[1];

  ptr_t<KDnode> root = *(ptr_t<KDnode> *)args;

  for(int i = 0; i < Config::num_objects; i++) {
    ptr_t<Object> optr = r_objects.template alloc<Object>();
    Object o;

    o.p[0] = 2*drand48() - 1;
    o.p[1] = 2*drand48() - 1;
    o.p[2] = 2*drand48() - 1;
    o.r = 0.05*drand48();

    // figure out which node this lives in
    ptr_t<KDnode> nptr = root;
    KDnode n;
    while(1) {
      n = r_tree.read(nptr);
      
      if(n.axis < 0) break; // leaf node
      
      if((o.p[n.axis] + o.r) < n.cutoff) {
	if(n.left) {
	  nptr = n.left;
	  continue;
	}
      }

      if((o.p[n.axis] - o.r) > n.cutoff) {
	if(n.right) {
	  nptr = n.right;
	  continue;
	}
      }

      // we straddle the split plane, so we get put in this node
      break;
    }

    o.next = n.objects;
    n.objects = optr;

    r_objects.write(optr, o);
    r_tree.write(nptr, n);
  }
}


int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;  

  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[TASKID_CREATE_TREE] = high_level_task_wrapper<CreateTreeResult,create_tree_task<AccessorGeneric> >;
  task_table[TASKID_ADD_OBJECTS] = high_level_task_wrapper<add_objects_task<AccessorGeneric> >;
  task_table[TASKID_FIND_ISECT] = high_level_task_wrapper<IntersectResult, find_isect_task<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-n")) {
      Config::num_tree_nodes = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-o")) {
      Config::num_objects = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-r")) {
      Config::num_rays = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) {
      Config::partition_size = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-v")) {
      Config::visible_partitions = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-s")) {
      Config::random_seed = atoi(argv[++i]);
      continue;
    }
  }

  Config::args_read = true;

  m.run();

  printf("Machine::run() finished!\n");

  return 0;
}

