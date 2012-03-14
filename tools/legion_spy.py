#!/usr/bin/python

import subprocess
import pygame, sys, os, shutil
from pygame.locals import *
from getopt import getopt

from parser import parse_log_file 

temp_dir = ".spy/"

def usage():
    print "Usage: "+ sys.argv[0] +" [-h (height in pixels)] [-w (width in pixels)] [-d (delta move size in pixels)] file_name"
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'d:h:w:')
    opts = dict(opts)
    if len(args) <> 1: 
        usage()

    height = int(opts.get('-h',640))
    width = int(opts.get('-w',480))
    delta = int(opts.get('-d',10))

    file_name = args[0]
    print 'Loading log file '+file_name+'...'
    log = parse_log_file(file_name)

    pygame.init()
    fpsClock = pygame.time.Clock()

    # Make a temporary directory
    print 'Generating images...'
    tree_images = log.print_trees(temp_dir)
    tree_surfaces = list()
    tree_origins = list()
    tree_files = list()
    for t in tree_images:
        tree_surf = pygame.image.load(t)
        tree_surfaces.append(tree_surf)
        tree_origins.append((0,0))
        tree_files.append(t)

    ctx_images = log.print_contexts(temp_dir)
    ctx_surfaces = list()
    ctx_origins = list()
    ctx_ids = list()
    ctx_files = list()
    for ctx_id,c in sorted(ctx_images.iteritems()):
        ctx_surf = pygame.image.load(c)
        ctx_surfaces.append(ctx_surf)
        ctx_origins.append((0,0))
        ctx_ids.append(ctx_id)
        ctx_files.append(c)

    event_img = log.print_event_graph(temp_dir)
    event_surf = pygame.image.load(event_img)
    event_origin = (0,0)

    print 'Initializing display...'

    showMode = 0
    numModes = 3
    currentOrigin = (0,0)
    currentTree = 0
    currentCtx = 0
    surface = pygame.display.set_mode((width,height))
    pygame.display.set_caption('Legion Spy')

    redColor = pygame.Color(255,0,0)
    greenColor = pygame.Color(0,255,0)
    blueColor = pygame.Color(0,0,255)
    whiteColor = pygame.Color(255,255,255)
    blackColor = pygame.Color(0,0,0)

    fontObj = pygame.font.Font(pygame.font.get_default_font(),32)

    movingLeft = False
    movingRight = False
    movingDown = False
    movingUp = False
        
    while True:
        justSet = False
        unSet = False
        # Handle all the events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYUP:
                if not justSet:
                    if event.key == K_h:
                        movingLeft = False
                    elif event.key == K_j:
                        movingDown = False
                    elif event.key == K_k:
                        movingUp = False
                    elif event.key == K_l:
                        movingRight = False
                else:
                    unSet = True
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == K_h: # move left
                    movingLeft = True
                    justSet = True
                elif event.key == K_j: # move down
                    movingDown = True    
                    justSet = True
                elif event.key == K_k: # move up
                    movingUp = True 
                    justSet = True
                elif event.key == K_l: # move right
                    movingRight = True
                    justSet = True
                elif event.key == K_r: # reset
                    if showMode == 0:
                        tree_origins[currentTree] = (0,0)
                    elif showMode == 1:
                        ctx_origins[currentCtx] = (0,0)
                    elif showMode == 2:
                        event_origin = (0,0)
                elif event.key == K_LEFT: # next picture 
                    if showMode == 0:
                        if currentTree == 0:
                            currentTree = len(tree_images)-1
                        else:
                            currentTree = currentTree-1
                    elif showMode == 1:
                        if currentCtx == 0:
                            currentCtx = len(ctx_images)-1
                        else:
                            currentCtx = currentCtx-1
                elif event.key == K_RIGHT: # previous picture
                    if showMode == 0:
                        currentTree = ((currentTree+1) % len(tree_images))
                    elif showMode == 1:
                        currentCtx = ((currentCtx+1) % len(ctx_images))
                elif event.key == K_DOWN:
                    showMode = (showMode + 1) % numModes
                elif event.key == K_UP:
                    if showMode == 0: 
                        showMode = numModes-1
                    else:
                        showMode = showMode-1
                elif event.key == K_p: #print the current image
                    if showMode == 0:
                        subprocess.call(['cp '+tree_files[currentTree]+' .'],shell=True)
                    elif showMode == 1:
                        subprocess.call(['cp '+ctx_files[currentCtx]+' .'],shell=True)
                    elif showMode == 2:
                        subprocess.call(['cp '+event_img+' .'],shell=True)
                    

        # Handle all moves
        if movingLeft:
            if showMode == 0:
                origin = tree_origins[currentTree]
                origin = (origin[0]+delta,origin[1])
                tree_origins[currentTree] = origin
            elif showMode == 1:
                origin = ctx_origins[currentCtx]
                origin = (origin[0]+delta,origin[1])
                ctx_origins[currentCtx] = origin
            elif showMode == 2:
                event_origin = (event_origin[0]+delta,event_origin[1])
        if movingDown: 
            if showMode == 0:
                origin = tree_origins[currentTree]
                origin = (origin[0],origin[1]-delta)
                tree_origins[currentTree] = origin
            elif showMode == 1:
                origin = ctx_origins[currentCtx]
                origin = (origin[0],origin[1]-delta)
                ctx_origins[currentCtx] = origin
            elif showMode == 2:
                event_origin = (event_origin[0],event_origin[1]-delta)
        if movingUp:
            if showMode == 0:
                origin = tree_origins[currentTree]
                origin = (origin[0],origin[1]+delta)
                tree_origins[currentTree] = origin
            elif showMode == 1:
                origin = ctx_origins[currentCtx]
                origin = (origin[0],origin[1]+delta)
                ctx_origins[currentCtx] = origin
            elif showMode == 2:
                event_origin = (event_origin[0],event_origin[1]+delta)
        if movingRight:
            if showMode == 0:
                origin = tree_origins[currentTree]
                origin = (origin[0]-delta,origin[1])
                tree_origins[currentTree] = origin
            elif showMode == 1:
                origin = ctx_origins[currentCtx]
                origin = (origin[0]-delta,origin[1])
                ctx_origins[currentCtx] = origin
            elif showMode == 2:
                event_origin = (event_origin[0]-delta,event_origin[1])

        if unSet:
            movingLeft = False
            movingDown = False
            movingUp = False
            movingRight = False

        # Render the scene
        surface.fill(whiteColor)

        if showMode == 0:
            tree_surf = tree_surfaces[currentTree]
            tree_origin = tree_origins[currentTree]
            surface.blit(tree_surf, tree_origin)
        elif showMode == 1:
            ctx_surf = ctx_surfaces[currentCtx]
            ctx_origin = ctx_origins[currentCtx]
            title = 'Context '+str(ctx_ids[currentCtx])
            msgSurfaceObj = fontObj.render(title,False,blackColor)
            msgRect = msgSurfaceObj.get_rect()
            surface.blit(ctx_surf, (ctx_origin[0],ctx_origin[1]+msgRect.height))
            surface.blit(msgSurfaceObj, (0,0))
        elif showMode == 2:
            surface.blit(event_surf, event_origin)

        pygame.display.update()
        fpsClock.tick(30)


if __name__ == "__main__":
    try:
        assert pygame.image.get_extended()
        os.mkdir(temp_dir)
        main()
        shutil.rmtree(temp_dir)
    except:
        # Remove the directory we created
        #shutil.rmtree(temp_dir)
        raise

